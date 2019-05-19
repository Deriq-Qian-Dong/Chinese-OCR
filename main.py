import cv2
import tkinter
import numpy as np
from crnn import util
from tkinter import *
import torch.utils.data
import tensorflow as tf
from crnn import dataset
import tkinter.filedialog
from crnn import keys_crnn
from crnn.models import crnn
from PIL import Image, ImageTk
from torch.autograd import Variable
from lib.fast_rcnn.test import _get_blobs
from tensorflow.python.platform import gfile
from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.text_connector.detectors import TextDetector
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from lib.text_connector.text_connect_cfg import Config as TextLineCfg


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


# input opencv 彩色照片list
# output string list 识别的文字
def crnn_batch(imglist):
    alphabet = keys_crnn.alphabet
    # print(len(alphabet))
    # input('\ninput:')
    converter = util.strLabelConverter(alphabet)
    # model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
    path = './crnn/samples/model_acc97.pth'
    model.load_state_dict(torch.load(path))
    strlist = []
    # print(model)
    for i in imglist:
        # print("cropimg size"+str(i.shape))
        img = Image.fromarray(np.array(i))
        image = img.convert('L')
        # print(image.size)
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        # print("width:" + str(w))
        transformer = dataset.resizeNormalize((w, 32))
        # image = transformer(image).cuda()
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        # print(preds.shape)
        _, preds = preds.max(2)
        # print(preds.shape)
        preds = preds.squeeze(1)
        preds = preds.transpose(-1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        sim_pred = sim_pred.lower()
        # print(sim_pred)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))
        strlist.append(sim_pred)
    return deletedot(strlist)

def process_boxes(img, boxes, scale):
    listoutput = []
    # print("img size"+str(img.shape))
    h = round(img.shape[0] / scale)
    w = round(img.shape[1] / scale)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        # print("更改前："+str(min_x) + "\t" +str(min_y) + "\t" + str(max_x) + "\t" +str(max_y))
        if min_y < 5:
            min_y = 0
        else:
            min_y = min_y - 5
        if (max_x + 5) > w:
            max_x = w
        else:
            max_x = max_x + 5
        if (max_y + 2) > h:
            max_y = h
        else:
            max_y = max_y + 2
        linestring = str(min_x) + "\t" + str(min_y) + "\t" + str(max_x) + "\t" + str(max_y)
        # print("更改后："+linestring)
        listoutput.append(linestring)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    return listoutput, img


# input: opencv 彩色照片
# output：string list 检测区域的四个坐标
def ctpn_single(img):
    cfg_from_file('./ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('./ctpn/data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    # print(img.shape)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    # print(boxes)
    strlist, img = process_boxes(img, boxes, scale)
    return strlist, img

# input opencv彩色照片，string list四个坐标的字符串
# output opencv list 照片list
def pt2img(strlist, img):
    # cv2.imshow("img", img)
    imglist = []
    for line in strlist:
        # print(line)
        pts = line.split("\t")
        img_crop = img[int(pts[1]):int(pts[3]), int(pts[0]):int(pts[2])]
        # print(img_crop.shape)
        # cv2.imshow(line, img_crop)
        imglist.append(img_crop)
    # while (1):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    return imglist


def deletedot(strlist):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    stroutput = []
    for line in strlist:
        line = cop.sub("", line)
        stroutput.append(line)
    return stroutput

def sort_strlist(strlist):
    t = [tt.split() for tt in strlist]
    t = np.array(t)
    t = t.astype(int)
    t = t.tolist()
    t = sorted(t, key=lambda x: sum([x[1]]))
    t = np.array(t).astype(str).tolist()
    return ["\t".join(tt) for tt in t]
# input opencv 单张彩色照片
# output string list识别的文字list
def ctpn_crnn_single(img):
    strlist, bounding_img = ctpn_single(img)
    strlist = sort_strlist(strlist)
    imglist = pt2img(strlist, img)
    result = crnn_batch(imglist)
    return result, bounding_img


def select_file():
    global file_name
    filename = tkinter.filedialog.askopenfilename()
    if filename != '':
        lb.config(text="您选择的文件是：" + filename)
        file_name = filename
    else:
        lb.config(text="您没有选择任何文件")
    try:
        im_path = file_name
        img = cv2.imread(im_path)
        result, bounding_img = ctpn_crnn_single(img)
        result = "\n".join(result)
        cv2.imwrite("tmp.jpg", bounding_img)
        pilImage = Image.open("tmp.jpg")
        tkImage = ImageTk.PhotoImage(image=pilImage)
        label.config(image=tkImage)
        label.image = tkImage
        if result:
            res.config(text="识别结果：" + result)
        else:
            res.config(text="识别结果：")
    except:
        res.config(text="识别结果：")


if __name__ == "__main__":
    file_name = ""
    root = tkinter.Tk(className="识别程序")
    root.geometry('800x500+800+500')
    lb = Label(root, text='')
    lb.pack()
    selectBtn = Button(root, text="选择图片", command=select_file)
    selectBtn.pack()
    res = Label(root, text='')
    res.pack()
    label = Label()
    label.pack()
    root.mainloop()
