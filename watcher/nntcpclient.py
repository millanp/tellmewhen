from __future__ import print_function
import os
import nnquery_pb2
import glob
import struct
import socket
import time
import sys
import cv2
import numpy as np

_this_file_dir = os.path.dirname(os.path.realpath(__file__))
opencv_feature_dir = '%s/../../data/opencv' % (_this_file_dir)
dnq1 = ':darknet:yolo9000:resize-img:0.24'
# dnq1 = ':darknet:yolo9000:resize-img:0.01'
LBL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def show_detection_result(msg, f, req, show_image=True):
    print('----------------------------------------------------------')
    print(str(msg))
    print('----------------------------------------------------------')
    if show_image:
        display_detection_result(msg, f, req)
    return


def get_wins(f, req):
    img = cv2.imread(f)
    whole_image_p = len(req.windows) == 0
    wins = (req.windows
            if not whole_image_p
            else [nnquery_pb2.NNRect(left=0,
                                     top=0,
                                     right=img.shape[1],
                                     bottom=img.shape[2])])

    return (img, wins, whole_image_p)


def display_detection_result(msg, f, req):
    img, wins, _ = get_wins(f, req)
    for ridx, rect in enumerate(wins):
        L, T = rect.left, rect.top
        dr = msg.detection_result[ridx]
        for i, rect in enumerate(dr.rects):
            cresult = dr.cresults[i]
            objectness = dr.objectness[i]
            txt = ('%s [id %d] prob:%0.2f [objness %0.2f]' %
                   (cresult.most_likely_class_name,
                    cresult.most_likely_class_id,
                    cresult.most_likely_class_prob,
                    objectness))
            l, t, r, b = L+rect.left, T+rect.top, L+rect.right, T+rect.bottom
            border_width = 2
            cv2.rectangle(img, (l, t), (r, b), (255, 0, 0), border_width)
            cv2.putText(img, txt, (l + 2, t + 2),
                        LBL_FONT, 0.5, (0, 0, 0), 1, 16)
    cv2.imshow('nnxs result', img)
    cv2.waitKey(0)
    return


def findfaces(imfile, cascf=None):
    cascf = cascf if cascf else ('%s/haarcascade_frontalface_default.xml' %
                                 (opencv_feature_dir))
    fcas = cv2.CascadeClassifier(cascf)
    img = cv2.imread(imfile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fcas.detectMultiScale(gray, 1.3, 5)
    # print('found %d faces in %s at %s' %
    #       (len(faces), imfile, str(faces)),
    #       file=sys.stderr)
    # return a list x,y,w,h of faces
    return faces


def preprocess(imfile, args):
    data = None
    action = args['preprocess']
    if action == 'findface':
        data = findfaces(imfile)
    else:
        print('Unknown preprocesing action %s' % (action), file=sys.stderr)
    return data


def send_message(sock, message):
    s = message.SerializeToString()
    packed_len = struct.pack('<L', len(s))
    sock.sendall(packed_len + s)


def socket_read_n(sock, n):
    """ Read exactly n bytes from the socket.
        Raise RuntimeError if the connection closed before
        n bytes were read.
    """
    buf = ''
    while n > 0:
        data = sock.recv(n)
        if data == '':
            raise RuntimeError('unexpected connection close')
        buf += data
        n -= len(data)
    return buf


def parse_args(argv):
    args = {}
    for arg in argv[1:]:
        assert arg[0] == '-', "arg should start with -"
        option, val = arg[1:].split('=')
        args[option] = val
    return args


def mkrequest(args):
    image = nnquery_pb2.NNImage(color=True)
    image.data = args['openimg'].read() # CHANGED: Open image file object passed in args

    query = args['query'] if 'query' in args else dnq1
    req = nnquery_pb2.NNRequest(reqid=217, query=query, image=image)

    # Sub-windows are not yet handled by the server
    # check if query should only apply to certain sub-windows of the image
    windows = []
    if 'preprocess' in args:
        if 'findface' == args['preprocess']:
            windows = preprocess(imfile_name, args)
    for x, y, w, h in windows:
        nnrect = req.windows.add()
        [nnrect.left, nnrect.top, nnrect.right, nnrect.bottom] = map(int,
                                                                     [x,
                                                                      y,
                                                                      x+w,
                                                                      y+h])

    return req, query, imfile_name


def remote_exec(req, args):
    msg = None
    HOST = args['host'] if 'host' in args else ''
    PORT = int(args['port']) if 'port' in args else 6868
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        beg = time.time()
        send_message(sock, req)
        len_buf = socket_read_n(sock, 4)
        msg_len = struct.unpack('<L', len_buf)[0]
        msg_buf = socket_read_n(sock, msg_len)
        msg = nnquery_pb2.NNResult()
        msg.ParseFromString(msg_buf)
        end = time.time()
    finally:
        sock.close()
    return (end-beg, msg)


def show_recognition_result(result_msg,
                            req):
    whole_image_p = len(req.windows) == 0
    wins = (req.windows
            if not whole_image_p
            else [nnquery_pb2.NNRect(left=0,
                                     top=0,
                                     right=req.img.shape[1],
                                     bottom=req.img.shape[2])])
    sys.stderr.write('[[[ msgid: %d \n' % result_msg.reqid)
    for i, win in enumerate(wins):
        cresult = result_msg.classification_result[i]
        sys.stderr.write('\t\t win: %s most likely clsid: %d prob: %f 0.3f\n' %
                         (str(win),
                          cresult.most_likely_class_id,
                          cresult.most_likely_class_prob))
    sys.stderr.write(']]]\n')
    return


def show_results(result_msg, args, query, imfile_name, delta_t, req):
    show_image = (bool(int(args['show_image']))
                  if 'show_image' in args else True)
    if 'yolo' in query or 'vgg_face' in query:
        show_detection_result(result_msg,
                              imfile_name,
                              req,
                              show_image=show_image)
        print('exec time %0.3f' % (delta_t * 1000.), file=sys.stderr)
    elif 'darknet:darknet' in query:
        print('msgid %d class res [%d %f]' %
              (result_msg.msgid,
               result_msg.classification_result.most_likely_class_id,
               result_msg.classification_result.most_likely_class_prob),
              file=sys.stderr)
        print('exec time %0.3f' % (delta_t * 1000.), file=sys.stderr)
        show_recognition_result(result_msg, req)
    else:
        print('----------- Message contents ---------', file=sys.stderr)
        print(str(result_msg), file=sys.stderr)
    return


def print_descriptor(dr):
    sys.stderr.write('\n==========================================\n')
    for id, f in enumerate(dr):
        if f != 0.:
            sys.stderr.write('-- %04d:%0.5f' % (id, f))
    sys.stderr.write('\n------------------------------------------\n')


def print_face_distances(results, reqs, args, namelen=5, paddedlen=20):
    name2desc = {}
    imgs = sorted(results.keys())

    for imgidx, img in enumerate(imgs):
        result = results[img]
        name = img.split('/')[-1].split('.')[-2]
        # associate file num with name
        name = '[%03d] %s' % (imgidx, name[:namelen])
        req = reqs[img]
        _, wins, whole_image_p = get_wins(img, req)
        for ridx, rect in enumerate(wins):
            L, T, R, B = rect.left, rect.top, rect.right, rect.bottom
            # associate rectange coords with name
            name1 = ('%s_L%d_T%d_R%d_B%d' % (name, L, T, R, B)
                     if not whole_image_p else name)
            dr = result.descriptor_result[ridx].descript
            # print_descriptor(dr)
            name2desc[name1] = np.array(dr)

    fmt = '%%%ds' % paddedlen
    fmt_float = '%%%d.5f' % paddedlen
    names = sorted(name2desc.keys())
    # list all images
    [print('[%3d] %s' % (imgidx, img), file=sys.stderr)
     for imgidx, img in enumerate(imgs)]
    # write the first row (headers)
    sys.stderr.write(fmt % '--')
    [sys.stderr.write(fmt % n) for n in names]
    sys.stderr.write('\n')
    # write out the contents of the table row-by-row
    for name in names:
        sys.stderr.write(fmt % name)
        dr = name2desc[name]
        for name1 in names:
            dr1 = name2desc[name1]
            dist = np.linalg.norm(dr-dr1)
            sys.stderr.write(fmt_float % dist)
        sys.stderr.write('\n')
    return


def process_image(args):
    # marshall argument values
    req, query, imfile_name = mkrequest(args)
    # send the request
    delta_t, result_msg = remote_exec(req, args)
    # render the results
    # show_results(result_msg, args, query, imfile_name, delta_t, req)
    return result_msg


def report_face_distances(args):
    img_glob = args['img']
    imgs = glob.glob(img_glob)
    results = {}
    reqs = {}
    for img in imgs:
        args['img'] = img
        req, query, imfile_name = mkrequest(args)
        delta_t, result_msg = remote_exec(req, args)
        results[img] = result_msg
        reqs[img] = req
    print_face_distances(results, reqs, args)
    return


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print("usage: python nntcpclient.py [jpeg]")
        exit()
    args = parse_args(sys.argv)
    print('args are %s' % str(args), file=sys.stderr)
    if 'report_face_distances' in args and args['report_face_distances'] != 0:
        report_face_distances(args)
    else:
        process_image(args)


# python nntcpclient.py ../../../src/c/darknet/data/eagle.jpg
# python nntcpclient.py -img=dog.jpg
# To send to remote server
# python nntcpclient.py -img=dog.jpg -host=23.96.10.248 -port=6868 -query=:darknet:yolo9000:resize-img:0.24 -show_image=0
# To send to local host server
# python nntcpclient.py -img=dog.jpg -port=6868 -query=:darknet:yolo9000:resize-img:0.24 -show_image=0
# python nntcpclient.py -img=ak.png -port=6868 -preprocess=findface -query=:caffe:vgg_face_caffe:resize-img:0.5 -show_image=0
# python nntcpclient.py -img=ak.png -port=6868 -preprocess=findface -query=:caffe:vgg_face_caffe+fc7:resize-img:0.5 -show_image=0
# python nntcpclient.py -report_face_distances=1 -img='../../data/faces/*.jpg' -port=6868 -preprocess=findface -query=:caffe:vgg_face_caffe+fc7:resize-img:0.5 -show_image=0
# python nntcpclient.py -host=23.96.10.248 -port=6868 -report_face_distances=1 -img='../../data/faces/*.jpg' -port=6868 -preprocess=findface -query=:caffe:vgg_face_caffe+fc7:resize-img:0.5 -show_image=0
