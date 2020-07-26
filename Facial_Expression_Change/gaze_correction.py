"""
視線補正を扱うモジュール.
コマンドラインで操作する場合は、追加でclickライブラリが必要.
"""

import math
import os

import cv2
import dlib
import numpy as np
import tensorflow as tf


_DEFAULT_DLIB_DAT_PATH = "data/shape_predictor_68_face_landmarks.dat"


class _TFUtil:
    @staticmethod
    def batch_norm(x, train_phase, name='bn_layer'):
        # with tf.variable_scope(name) as scope:
        batch_norm = tf.layers.batch_normalization(
            inputs=x,
            momentum=0.9, epsilon=1e-5,
            center=True, scale=True,
            training=train_phase,
            name=name
        )
        return batch_norm

    @staticmethod
    def cnn_blk(inputs, filters, kernel_size, phase_train, name='cnn_blk'):
        with tf.variable_scope(name) as scope:
            cnn = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding="same", activation=None, use_bias=False, name="cnn")
            act = tf.nn.relu(cnn, name="act")
            ret = _TFUtil.batch_norm(act, phase_train)
            return ret

    @staticmethod
    def dnn_blk(inputs, nodes, name='dnn_blk'):
        with tf.variable_scope(name) as scope:
            dnn = tf.layers.dense(inputs=inputs, units=nodes, activation=None, name="dnn")
            ret = tf.nn.relu(dnn, name="act")
            return ret


class _Transformation:

    @classmethod
    def repeat(cls, x, num_repeats):
        with tf.name_scope("repeat"):
            ones = tf.ones((1, num_repeats), dtype='int32')
            x = tf.reshape(x, shape=(-1, 1))
            x = tf.matmul(x, ones)
            return tf.reshape(x, [-1])

    @classmethod
    def interpolate(cls, image, x, y, output_size):
        with tf.name_scope("interpolate"):
            batch_size = tf.shape(image)[0]
            height = tf.shape(image)[1]
            width = tf.shape(image)[2]
            num_channels = tf.shape(image)[3]

            x = tf.cast(x, dtype='float32')
            y = tf.cast(y, dtype='float32')

            height_float = tf.cast(height, dtype='float32')
            width_float = tf.cast(width, dtype='float32')

            output_height = output_size[0]
            output_width = output_size[1]

            x = .5 * (x + 1.0) * (width_float)
            y = .5 * (y + 1.0) * (height_float)

            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            max_y = tf.cast(height - 1, dtype='int32')
            max_x = tf.cast(width - 1, dtype='int32')
            zero = tf.zeros([], dtype='int32')

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            flat_image_dimensions = height * width
            pixels_batch = tf.range(batch_size) * flat_image_dimensions
            flat_output_dimensions = output_height * output_width
            base = cls.repeat(pixels_batch, flat_output_dimensions)
            base_y0 = base + y0 * width
            base_y1 = base + y1 * width
            indices_a = base_y0 + x0
            indices_b = base_y1 + x0
            indices_c = base_y0 + x1
            indices_d = base_y1 + x1

            flat_image = tf.reshape(image, shape=(-1, num_channels))
            flat_image = tf.cast(flat_image, dtype='float32')
            pixel_values_a = tf.gather(flat_image, indices_a)
            pixel_values_b = tf.gather(flat_image, indices_b)
            pixel_values_c = tf.gather(flat_image, indices_c)
            pixel_values_d = tf.gather(flat_image, indices_d)

            x0 = tf.cast(x0, 'float32')
            x1 = tf.cast(x1, 'float32')
            y0 = tf.cast(y0, 'float32')
            y1 = tf.cast(y1, 'float32')

            area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
            area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
            area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
            area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
            output = tf.add_n([area_a * pixel_values_a,
                               area_b * pixel_values_b,
                               area_c * pixel_values_c,
                               area_d * pixel_values_d])
            return output

    @classmethod
    def meshgrid(cls, height, width):
        with tf.name_scope("meshgrid"):
            y_linspace = tf.linspace(-1., 1., height)
            x_linspace = tf.linspace(-1., 1., width)
            x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
            y_coordinates = tf.expand_dims(tf.reshape(y_coordinates, [-1]), 0)
            x_coordinates = tf.expand_dims(tf.reshape(x_coordinates, [-1]), 0)
            indices_grid = tf.concat([x_coordinates, y_coordinates], 0)
            return indices_grid

    @classmethod
    def apply_transformation(cls, flows, img, num_channels):
        with tf.name_scope("apply_transformation"):
            batch_size = tf.shape(img)[0]
            height = tf.shape(img)[1]
            width = tf.shape(img)[2]
            # num_channels = tf.shape(img)[3]
            output_size = (height, width)
            flow_channels = tf.shape(flows)[3]

            flows = tf.reshape(tf.transpose(flows, [0, 3, 1, 2]), [batch_size, flow_channels, height * width])

            indices_grid = cls.meshgrid(height, width)

            transformed_grid = tf.add(flows, indices_grid)
            x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_image = cls.interpolate(img, x_s_flatten, y_s_flatten, (height, width))

            transformed_image = tf.reshape(transformed_image, [batch_size, height, width, num_channels])
            return transformed_image


class FLX:
    img_crop = 3

    @classmethod
    def gen_agl_map(cls, inputs, height, width, feature_dims):
        with tf.name_scope("gen_agl_map"):
            batch_size = tf.shape(inputs)[0]
            ret = tf.reshape(tf.tile(inputs, tf.constant([1, height * width])), [batch_size, height, width, feature_dims])
            return ret

    @classmethod
    def encoder(cls, inputs, height, width, tar_dim):
        with tf.variable_scope('encoder'):
            dnn_blk_0 = _TFUtil.dnn_blk(inputs, 16, name='dnn_blk_0')
            dnn_blk_1 = _TFUtil.dnn_blk(dnn_blk_0, 16, name='dnn_blk_1')
            dnn_blk_2 = _TFUtil.dnn_blk(dnn_blk_1, tar_dim, name='dnn_blk_2')
            agl_map = cls.gen_agl_map(dnn_blk_2, height, width, tar_dim)
            return agl_map

    @classmethod
    def apply_lcm(cls, batch_img, light_weight):
        with tf.name_scope('apply_lcm'):
            img_wgts, pal_wgts = tf.split(light_weight, [1, 1], 3)
            img_wgts = tf.tile(img_wgts, [1, 1, 1, 3])
            pal_wgts = tf.tile(pal_wgts, [1, 1, 1, 3])
            palette = tf.ones(tf.shape(batch_img), dtype=tf.float32)
            ret = tf.add(tf.multiply(batch_img, img_wgts), tf.multiply(palette, pal_wgts))
            return ret

    @classmethod
    def trans_module(cls, inputs, structures, phase_train, name="trans_module"):
        with tf.variable_scope(name) as scope:
            cnn_blk_0 = _TFUtil.cnn_blk(inputs, structures['depth'][0], structures['filter_size'][0], phase_train, name='cnn_blk_0')
            cnn_blk_1 = _TFUtil.cnn_blk(cnn_blk_0, structures['depth'][1], structures['filter_size'][1], phase_train, name='cnn_blk_1')
            cnn_blk_2 = _TFUtil.cnn_blk(tf.concat([cnn_blk_0, cnn_blk_1], axis=3), structures['depth'][2], structures['filter_size'][2], phase_train,
                                        name='cnn_blk_2')
            cnn_blk_3 = _TFUtil.cnn_blk(tf.concat([cnn_blk_0, cnn_blk_1, cnn_blk_2], axis=3), structures['depth'][3], structures['filter_size'][3],
                                        phase_train, name='cnn_blk_3')
            cnn_4 = tf.layers.conv2d(inputs=cnn_blk_3, filters=structures['depth'][4], kernel_size=structures['filter_size'][4], padding="same",
                                     activation=None, use_bias=False, name="cnn_4")
            return cnn_4

    @classmethod
    def lcm_module(cls, inputs, structures, phase_train, name="lcm_module"):
        with tf.variable_scope(name) as scope:
            cnn_blk_0 = _TFUtil.cnn_blk(inputs, structures['depth'][0], structures['filter_size'][0], phase_train, name='cnn_blk_0')
            cnn_blk_1 = _TFUtil.cnn_blk(cnn_blk_0, structures['depth'][1], structures['filter_size'][1], phase_train, name='cnn_blk_1')
            cnn_2 = tf.layers.conv2d(inputs=cnn_blk_1, filters=structures['depth'][2], kernel_size=structures['filter_size'][2], padding="same",
                                     activation=None, use_bias=False, name='cnn_2')
            lcm_map = tf.nn.softmax(cnn_2)
            return lcm_map

    @classmethod
    def inference(cls, input_img, input_fp, input_agl, phase_train, height, width, encoded_agl_dim):
        """Build the Deepwarp model.
        Args: images, anchors_map of eye, angle
        Returns: lcm images
        """
        corse_layer = {'depth': (32, 64, 64, 32, 16), 'filter_size': ([5, 5], [3, 3], [3, 3], [3, 3], [1, 1])}
        fine_layer = {'depth': (32, 64, 32, 16, 4), 'filter_size': ([5, 5], [3, 3], [3, 3], [3, 3], [1, 1])}
        lcm_layer = {'depth': (8, 8, 2), 'filter_size': ([3, 3], [3, 3], [1, 1])}

        with tf.variable_scope('warping_model'):
            agl_map = cls.encoder(input_agl, height, width, encoded_agl_dim)
            igt_inputs = tf.concat([input_img, input_fp, agl_map], axis=3)

            with tf.variable_scope('warping_module'):
                '''coarse module'''
                resized_igt_inputs = tf.layers.average_pooling2d(inputs=igt_inputs, pool_size=[2, 2], strides=2, padding='same')
                cours_raw = cls.trans_module(resized_igt_inputs, corse_layer, phase_train, name='coarse_level')
                cours_act = tf.nn.tanh(cours_raw)
                coarse_resize = tf.image.resize_images(cours_act, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                coarse_out = tf.layers.average_pooling2d(inputs=coarse_resize, pool_size=[2, 2], strides=1, padding='same')
                '''fine module'''
                fine_input = tf.concat([igt_inputs, coarse_out], axis=3, name='fine_input')
                fine_out = cls.trans_module(fine_input, fine_layer, phase_train, name='fine_level')
                flow_raw, lcm_input = tf.split(fine_out, [2, 2], 3)

            flow = tf.nn.tanh(flow_raw)
            cfw_img = _Transformation.apply_transformation(flows=flow, img=input_img, num_channels=3)
            '''lcm module'''
            lcm_map = cls.lcm_module(lcm_input, lcm_layer, phase_train, name="lcm_module")
            img_pred = cls.apply_lcm(batch_img=cfw_img, light_weight=lcm_map)

            return img_pred, flow_raw, lcm_map

    @classmethod
    def dist_loss(cls, y_pred, y_, method="MAE"):
        img_crop = 3
        with tf.variable_scope('img_dist_loss'):
            loss = 0
            if (method == "L2"):
                loss = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_), axis=3, keep_dims=True))
            elif (method == "MAE"):
                loss = tf.abs(y_pred - y_)
            loss = loss[:, img_crop:(-1) * img_crop, img_crop:(-1) * img_crop, :]
            loss = tf.reduce_sum(loss, axis=[1, 2, 3])
            return tf.reduce_mean(loss, axis=0)

    @classmethod
    def TVloss(cls, inputs):
        with tf.variable_scope('TVloss'):
            dinputs_dx = inputs[:, :-1, :, :] - inputs[:, 1:, :, :]
            dinputs_dy = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
            dinputs_dx = tf.pad(dinputs_dx, [[0, 0], [0, 1], [0, 0], [0, 0]], "CONSTANT")
            dinputs_dy = tf.pad(dinputs_dy, [[0, 0], [0, 0], [0, 1], [0, 0]], "CONSTANT")
            tot_var = tf.add(tf.abs(dinputs_dx), tf.abs(dinputs_dy))
            tot_var = tf.reduce_sum(tot_var, axis=3, keep_dims=True)
            return tot_var

    @classmethod
    def TVlosses(cls, eye_mask, ori_img, flow, lcm_map):
        with tf.variable_scope('TVlosses'):
            # eyeball_TVloss
            # calculate TV (dFlow(p)/dx  + dFlow(p)/dy)
            TV_flow = cls.TVloss(flow)
            # calculate the (1-D(p))
            img_gray = tf.reduce_mean(ori_img, axis=3, keep_dims=True)
            ones = tf.ones(shape=tf.shape(img_gray))
            bright = ones - img_gray
            # calculate the F_e(p)
            eye_mask = tf.expand_dims(eye_mask, axis=3)
            weights = tf.multiply(bright, eye_mask)
            TV_eye = tf.multiply(weights, TV_flow)

            # eyelid_TVloss
            lid_mask = ones - eye_mask
            TV_lid = tf.multiply(lid_mask, TV_flow)

            TV_eye = tf.reduce_sum(TV_eye, axis=[1, 2, 3])
            TV_lid = tf.reduce_sum(TV_lid, axis=[1, 2, 3])

            # lcm_map loss
            dist2cent = cls.center_weight(tf.shape(lcm_map), base=0.005, boundary_penalty=3.0)
            TV_lcm = dist2cent * cls.TVloss(lcm_map)
            TV_lcm = tf.reduce_sum(TV_lcm, axis=[1, 2, 3])

            return tf.reduce_mean(TV_eye, axis=0), tf.reduce_mean(TV_lid, axis=0), tf.reduce_mean(TV_lcm, axis=0)

    @classmethod
    def center_weight(cls, shape, base=0.005, boundary_penalty=3.0):
        with tf.variable_scope('center_weight'):
            temp = boundary_penalty - base
            x = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0, shape[1])), 8)
            y = tf.pow(tf.abs(tf.lin_space(-1.0, 1.0, shape[2])), 8)
            X, Y = tf.meshgrid(y, x)
            X = tf.expand_dims(X, axis=2)
            Y = tf.expand_dims(Y, axis=2)
            dist2cent = temp * tf.sqrt(tf.reduce_sum(tf.square(tf.concat([X, Y], axis=2)), axis=2)) + base
            dist2cent = tf.expand_dims(tf.tile(tf.expand_dims(dist2cent, axis=0), [shape[0], 1, 1]), axis=3)
            return dist2cent

    @classmethod
    def lcm_adj(cls, lcm_wgt):
        dist2cent = cls.center_weight(tf.shape(lcm_wgt), base=0.005, boundary_penalty=3.0)
        with tf.variable_scope('lcm_adj'):
            _, loss = tf.split(lcm_wgt, [1, 1], 3)
            loss = tf.reduce_sum(tf.abs(loss) * dist2cent, axis=[1, 2, 3])
            return tf.reduce_mean(loss, axis=0)

    @classmethod
    def loss(cls, img_pred, img_, eye_mask, input_img, flow, lcm_wgt):
        with tf.variable_scope('losses'):
            loss_img = cls.dist_loss(img_pred, img_, method="L2")

            loss_eyeball, loss_eyelid, loss_lcm = cls.TVlosses(eye_mask, input_img, flow, lcm_wgt)
            loss_lcm_adj = cls.lcm_adj(lcm_wgt)

            losses = loss_img + loss_eyeball + loss_eyelid + loss_lcm_adj + loss_lcm
            tf.add_to_collection('losses', losses)
            return tf.add_n(tf.get_collection('losses'), name='total_loss'), loss_img


class FocalLengthCalibrator:
    """
    カメラから50cm離れて、キャリブレーションする
    """

    def __init__(self, interpupillary_distance=6.3, detector=None, predictor=None, video_size=(640, 480), face_detect_size=(640, 480)):

        # カメラから顔までの距離
        self.distance_from_camera = 50

        # 瞳孔間距離
        self.interpupillary_distance = interpupillary_distance

        self.video_size = video_size

        if detector is None:
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = detector

        if predictor is None:
            self.predictor = dlib.shape_predictor(_DEFAULT_DLIB_DAT_PATH)
        else:
            self.predictor = predictor

        self.face_detect_size = face_detect_size

        self._focal_length = None

    def get_eye_pos(self, shape, pos="L"):
        if pos == "R":
            eye_left_corner_id = 36
            eye_right_corner_id = 39

        elif pos == "L":
            eye_left_corner_id = 42
            eye_right_corner_id = 45

        else:
            print("Error: Wrong Eye")

        # 論文ではar
        eye_corner_r = shape.part(eye_right_corner_id)
        # 論文ではal
        eye_corner_l = shape.part(eye_left_corner_id)

        # 目の中心座標
        eye_cx = (eye_corner_r.x + eye_corner_l.x) * 0.5
        eye_cy = (eye_corner_r.y + eye_corner_l.y) * 0.5
        eye_center = [eye_cx, eye_cy]

        # 片目の幅
        eye_len = np.absolute(eye_corner_r.x - eye_corner_l.x)

        # 論文に書いてある詳細と実装が少し違う

        # 多くの目を観察し、目玉の幅:目の幅は3:4とする. (論文より)
        # 片目玉の幅
        bx_d5w = eye_len * 3 / 4

        # 目領域のROIの高さ:
        bx_h = 1.5 * bx_d5w
        # 目の中心のY座標から、目ROIの上辺と下辺までの距離
        # 上まぶたの方が下まぶたより良く動くため 6/12ではなく、7/12と5/12にしている(論文より)
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        E_TL = (int(eye_cx - bx_d5w), int(eye_cy - sft_up))
        E_RB = (int(eye_cx + bx_d5w), int(eye_cy + sft_low))
        return eye_center, E_TL, E_RB

    def start(self):
        """
        Start capturing you faces, push k if you have already placed you head about 50 cm
        :return:
        """

        vs = cv2.VideoCapture(0)

        while True:
            ret, recv_frame = vs.read()

            gray = cv2.cvtColor(recv_frame, cv2.COLOR_BGR2GRAY)
            face_detect_gray = cv2.resize(gray, (self.face_detect_size[0], self.face_detect_size[1]))
            # Detect the facial landmarks
            detections = self.detector(face_detect_gray, 0)
            x_ratio = self.video_size[0] / self.face_detect_size[0]
            y_ratio = self.video_size[1] / self.face_detect_size[1]
            for k, bx in enumerate(detections):
                target_bx = dlib.rectangle(left=int(bx.left() * x_ratio), right=int(bx.right() * x_ratio),
                                           top=int(bx.top() * y_ratio), bottom=int(bx.bottom() * y_ratio))
                shape = self.predictor(gray, target_bx)
                # get eye
                LE_center, L_E_TL, L_E_RB = self.get_eye_pos(shape, pos="L")
                RE_center, R_E_TL, R_E_RB = self.get_eye_pos(shape, pos="R")

                self._focal_length = int(
                    np.sqrt((LE_center[0] - RE_center[0]) ** 2 + (LE_center[1] - RE_center[1]) ** 2) * self.distance_from_camera / self.interpupillary_distance)
                cv2.rectangle(recv_frame,
                              (self.video_size[0] - 150, 0), (self.video_size[0], 40),
                              (255, 255, 255), -1
                              )
                cv2.putText(recv_frame,
                            'f:' + str(self._focal_length),
                            (self.video_size[0] - 140, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                #         cv2.line(recv_frame, (int(LE_center[0]),int(LE_center[1])), (int(RE_center[0]),int(RE_center[1])), (0,0,255))

                # eye region
                cv2.rectangle(recv_frame,
                              (L_E_TL[0], L_E_TL[1]), (L_E_RB[0], L_E_RB[1]),
                              (0, 0, 255), 1
                              )
                cv2.rectangle(recv_frame,
                              (R_E_TL[0], R_E_TL[1]), (R_E_RB[0], R_E_RB[1]),
                              (0, 0, 255), 1
                              )
                cv2.circle(recv_frame, (int(LE_center[0]), int(LE_center[1])), 2, (0, 255, 0), -1)
                cv2.circle(recv_frame, (int(RE_center[0]), int(RE_center[1])), 2, (0, 255, 0), -1)
                for i in range(68):
                    cv2.circle(recv_frame, (shape.part(i).x, shape.part(i).y), 2, (0, 0, 255), -1)

            cv2.imshow("Calibration", recv_frame)
            k = cv2.waitKey(10)
            if k == ord('q'):
                vs.release()
                cv2.destroyAllWindows()
                break
            else:
                pass

        print("focal length is ", self._focal_length)

    @property
    def focal_length(self):
        return self._focal_length


class GazeCorrector:
    """
    視線補正を行うクラス.
    """

    def __init__(self, dlib_dat_path, model_dir, screen_size_cm, screen_size_pt, app_window_rect, video_size, camera_pos_cm, interpupillary_distance_cm=6.3,
                 focal_length=315, face_detect_size=(640, 480)):
        # 顔検出器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_dat_path)

        # 学習済みモデルのパラメータチェックポイントがあるディレクトリパス
        self._model_dir_L = os.path.join(model_dir, 'L') + os.sep
        self._model_dir_R = os.path.join(model_dir, 'R') + os.sep

        # Screenサイズ (cm) : Ps
        self.screen_w_cm = screen_size_cm[0]
        self.screen_h_cm = screen_size_cm[1]

        # ディスプレイサイズ (pt)
        self.screen_w_pt = screen_size_pt[0]
        self.screen_h_pt = screen_size_pt[1]

        # 相手の顔のウィンドウのRectを取得する(ディスプレイ座標系 pt): win32gui等を利用 (とりあえず固定値)
        self.app_window_x = app_window_rect[0]
        self.app_window_y = app_window_rect[1]
        self.app_window_w = app_window_rect[2]
        self.app_window_h = app_window_rect[3]

        # ビデオフレームのサイズ (pt)
        self.video_w = video_size[0]
        self.video_h = video_size[1]

        # 瞳孔間距離
        self.interpupillary_distance_cm = interpupillary_distance_cm

        # Pc: カメラの座標(cm)
        self.Pc_x = camera_pos_cm[0]
        self.Pc_y = camera_pos_cm[1]
        self.Pc_z = camera_pos_cm[2]
        self.Pc = camera_pos_cm

        # focal length
        self.focal_length = focal_length

        self.face_detect_size = face_detect_size

        # 画像処理対象とする目のCrop画像サイズ
        self.size_I = (64, 48)

        # Initial value
        self.Rw = [0, 0]

        # Pe: 自分のグローバル座標 cm
        # 初期値
        self.Pe = [self.Pc[0], self.Pc[1], -60]  # H,V,D

        # モデルに入力する目の画像
        input_eye_img_width = 64
        input_eye_img_height = 48
        input_eye_img_channel = 3
        eye_landmarks_counts = 6
        eye_landmarks_dim = eye_landmarks_counts * 2  # (x, y)の分

        agl_dim = 2
        encoded_agl_dim = 16

        # load model to gpu
        print("Loading model of [L] eye to GPU")
        with tf.Graph().as_default() as g:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.LE_input_img = tf.placeholder(tf.float32, [None, input_eye_img_height, input_eye_img_width, input_eye_img_channel], name="input_img")
                self.LE_input_fp = tf.placeholder(tf.float32, [None, input_eye_img_height, input_eye_img_width, eye_landmarks_dim], name="input_fp")
                self.LE_input_ang = tf.placeholder(tf.float32, [None, agl_dim], name="input_ang")
                self.LE_phase_train = tf.placeholder(tf.bool, name='phase_train')  # a bool for batch_normalization

            self.LE_img_pred, _, _ = FLX.inference(
                self.LE_input_img,
                self.LE_input_fp,
                self.LE_input_ang,
                self.LE_phase_train,
                input_eye_img_height,
                input_eye_img_width,
                encoded_agl_dim)

            # split modle here
            self.L_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=g)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self._model_dir_L)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.L_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

        print("Loading model of [R] eye to GPU")
        with tf.Graph().as_default() as g2:
            # define placeholder for inputs to network
            with tf.name_scope('inputs'):
                self.RE_input_img = tf.placeholder(tf.float32, [None, input_eye_img_height, input_eye_img_width, input_eye_img_channel], name="input_img")
                self.RE_input_fp = tf.placeholder(tf.float32, [None, input_eye_img_height, input_eye_img_width, eye_landmarks_dim], name="input_fp")
                self.RE_input_ang = tf.placeholder(tf.float32, [None, agl_dim], name="input_ang")
                self.RE_phase_train = tf.placeholder(tf.bool, name='phase_train')  # a bool for batch_normalization

            self.RE_img_pred, _, _ = FLX.inference(
                self.RE_input_img,
                self.RE_input_fp,
                self.RE_input_ang,
                self.RE_phase_train,
                input_eye_img_height,
                input_eye_img_width,
                encoded_agl_dim)

            # split modle here
            self.R_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=g2)
            # load model
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(self._model_dir_R)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(self.R_sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')

    def draw_parameters(self, frame, fig_alpha, fig_eye_pos, fig_R_w):
        cv2.rectangle(frame,
                      (self.video_w - 150, 0), (self.video_w, 55),
                      (255, 255, 255), -1
                      )
        cv2.putText(frame,
                    'Eye:[' + str(int(fig_eye_pos[0])) + ',' + str(int(fig_eye_pos[1])) + ',' + str(int(fig_eye_pos[2])) + ']',
                    (self.video_w - 140, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'alpha:[V=' + str(int(fig_alpha[0])) + ',H=' + str(int(fig_alpha[1])) + ']',
                    (self.video_w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame,
                    'R_w:[' + str(int(fig_R_w[0])) + ',' + str(int(fig_R_w[1])) + ']',
                    (self.video_w - 140, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        return frame

    def create_eye_roi_anchormap(self, frame, shape, pos="L", size_I=None):
        if size_I is None:
            size_I = self.size_I

        if pos == "R":
            eye_left_corner_id = 36
            eye_right_corner_id = 39
            eye_contour_indices = [36, 37, 38, 39, 40, 41]
        elif pos == "L":
            eye_left_corner_id = 42
            eye_right_corner_id = 45
            eye_contour_indices = [45, 44, 43, 42, 47, 46]
        else:
            raise ValueError("invalid pos. specify L or R")

        # 論文ではar
        eye_corner_r = shape.part(eye_right_corner_id)
        # 論文ではal
        eye_corner_l = shape.part(eye_left_corner_id)

        # 目の中心座標
        eye_cx = (eye_corner_r.x + eye_corner_l.x) * 0.5
        eye_cy = (eye_corner_r.y + eye_corner_l.y) * 0.5
        eye_center = [eye_cx, eye_cy]

        # 片目の幅
        eye_len = np.absolute(eye_corner_r.x - eye_corner_l.x)

        # 論文に書いてある詳細と実装が少し違う

        # 多くの目を観察し、目玉の幅:目の幅は3:4とする. (論文より)
        # 片目玉の幅
        bx_d5w = eye_len * 3 / 4

        # 目領域のROIの高さ:
        bx_h = 1.5 * bx_d5w
        # 目の中心のY座標から、目ROIの上辺と下辺までの距離
        # 上まぶたの方が下まぶたより良く動くため 6/12ではなく、7/12と5/12にしている(論文より)
        sft_up = bx_h * 7 / 12
        sft_low = bx_h * 5 / 12

        # 目領域のROI
        img_eye = frame[int(eye_cy - sft_up):int(eye_cy + sft_low), int(eye_cx - bx_d5w):int(eye_cx + bx_d5w)]

        # オリジナルの目領域のROIのサイズ
        org_eye_roi_size = (img_eye.shape[0], img_eye.shape[1])

        # オリジナルの目領域のROIのLeftTop座標
        org_eye_roi_lt = (int(eye_cy - sft_up), int(eye_cx - bx_d5w))  # (y,x)

        # 目領域の画像をリサイズする
        img_eye = cv2.resize(img_eye, size_I)

        # create anchor maps
        ach_map = []
        for i, landmark_id in enumerate(eye_contour_indices):

            # リサイズされた
            resize_x = int((shape.part(landmark_id).x - org_eye_roi_lt[1]) * size_I[0] / org_eye_roi_size[1])
            resize_y = int((shape.part(landmark_id).y - org_eye_roi_lt[0]) * size_I[1] / org_eye_roi_size[0])

            # y
            ach_map_y = np.expand_dims(np.expand_dims(np.arange(0, size_I[1]) - resize_y, axis=1), axis=2)
            ach_map_y = np.tile(ach_map_y, [1, size_I[0], 1])

            # x
            ach_map_x = np.expand_dims(np.expand_dims(np.arange(0, size_I[0]) - resize_x, axis=0), axis=2)
            ach_map_x = np.tile(ach_map_x, [size_I[1], 1, 1])

            if i == 0:
                ach_map = np.concatenate((ach_map_x, ach_map_y), axis=2)
            else:
                ach_map = np.concatenate((ach_map, ach_map_x, ach_map_y), axis=2)

        return img_eye / 255, ach_map, eye_center, org_eye_roi_size, org_eye_roi_lt

    def estimate_shift_angles(self, L_eye_center, R_eye_center):

        # そのウィンドウ内における顔の中心位置 (pt) (とりあえず中心にあると仮定し固定値にする) (x, y)
        peer_face_relative_x = int(self.app_window_w / 2)
        peer_face_relative_y = int(self.app_window_h / 2)

        # ディスプレイ座標系における相手の顔の中心座標 (x, y)
        peer_face_x = self.app_window_x + peer_face_relative_x
        peer_face_y = self.app_window_y + peer_face_relative_y
        peer_face_pos = (peer_face_x, peer_face_y)

        # Pw: 相手の顔の中心座標（グローバル座標）を計算する (cm)
        # グローバル座標はディスプレイの中心を(0,0,0)とする
        # pixelでの比率を使って、実寸(cm)を算出する
        Pw_x = self.screen_w_cm * (peer_face_x - self.screen_w_pt / 2) / self.screen_w_pt
        Pw_y = self.screen_h_cm * (peer_face_y - self.screen_h_pt / 2) / self.screen_h_pt
        Pw_z = 0
        Pw = (Pw_x, Pw_y, Pw_z)

        # Pe: 自分の座標
        interpupillary_distance_pt = np.sqrt((L_eye_center[0] - R_eye_center[0]) ** 2 + (L_eye_center[1] - R_eye_center[1]) ** 2)
        self.Pe[2] = - self.focal_length * self.interpupillary_distance_cm / interpupillary_distance_pt

        # x-axis needs flip
        self.Pe[0] = -np.abs(self.Pe[2]) * (L_eye_center[0] + R_eye_center[0] - self.video_w) / (2 * self.focal_length) + self.Pc_x
        self.Pe[1] = +np.abs(self.Pe[2]) * (L_eye_center[1] + R_eye_center[1] - self.video_h) / (2 * self.focal_length) + self.Pc_y

        # alphaを算出する
        a_w2z_x = math.degrees(math.atan((Pw[0] - self.Pe[0]) / (Pw[2] - self.Pe[2])))
        a_w2z_y = math.degrees(math.atan((Pw[1] - self.Pe[1]) / (Pw[2] - self.Pe[2])))
        a_z2c_x = math.degrees(math.atan((self.Pe[0] - self.Pc[0]) / (self.Pc[2] - self.Pe[2])))
        a_z2c_y = math.degrees(math.atan((self.Pe[1] - self.Pc[1]) / (self.Pc[2] - self.Pe[2])))
        alpha = [int(a_w2z_y + a_z2c_y), int(a_w2z_x + a_z2c_x)]  # (V,H)

        return alpha, self.Pe, peer_face_pos

    def correct(self, frame, gray, detections, pixel_cut=(3, 4), size_I=(64, 48), draw_param=False):
        alpha_w2c = [0, 0]
        x_ratio = self.video_w / self.face_detect_size[0]
        y_ratio = self.video_h / self.face_detect_size[1]

        LE_M_A = []
        RE_M_A = []
        p_e = [0, 0]
        R_w = [0, 0]

        for k, bx in enumerate(detections):
            # Get facial landmarks
            target_bx = dlib.rectangle(left=int(bx.left() * x_ratio), right=int(bx.right() * x_ratio),
                                       top=int(bx.top() * y_ratio), bottom=int(bx.bottom() * y_ratio))
            shape = self.predictor(gray, target_bx)

            # get eye
            L_eye_img, LE_M_A, L_eye_center, L_org_eye_roi_size, L_org_eye_roi_lt = self.create_eye_roi_anchormap(frame, shape, pos="L", size_I=size_I)
            R_eye_img, RE_M_A, R_eye_center, R_org_eye_roi_size, R_org_eye_roi_lt = self.create_eye_roi_anchormap(frame, shape, pos="R", size_I=size_I)

            # shifting angles estimator
            alpha_w2c, p_e, R_w = self.estimate_shift_angles(L_eye_center, R_eye_center)

            # gaze redirection
            # Left Eye
            LE_infer_img = self.L_sess.run(self.LE_img_pred, feed_dict={
                self.LE_input_img: np.expand_dims(L_eye_img, axis=0),
                self.LE_input_fp: np.expand_dims(LE_M_A, axis=0),
                self.LE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.LE_phase_train: False
            })
            LE_infer = cv2.resize(LE_infer_img.reshape(size_I[1], size_I[0], 3), (L_org_eye_roi_size[1], L_org_eye_roi_size[0]))

            # Right Eye
            RE_infer_img = self.R_sess.run(self.RE_img_pred, feed_dict={
                self.RE_input_img: np.expand_dims(R_eye_img, axis=0),
                self.RE_input_fp: np.expand_dims(RE_M_A, axis=0),
                self.RE_input_ang: np.expand_dims(alpha_w2c, axis=0),
                self.RE_phase_train: False
            })
            RE_infer = cv2.resize(RE_infer_img.reshape(size_I[1], size_I[0], 3), (R_org_eye_roi_size[1], R_org_eye_roi_size[0]))

            # 目を置き換える
            frame[(L_org_eye_roi_lt[0] + pixel_cut[0]):(L_org_eye_roi_lt[0] + L_org_eye_roi_size[0] - pixel_cut[0]),
            (L_org_eye_roi_lt[1] + pixel_cut[1]):(L_org_eye_roi_lt[1] + L_org_eye_roi_size[1] - pixel_cut[1])] = LE_infer[pixel_cut[0]:(-1 * pixel_cut[0]),
                                                                                                                 pixel_cut[1]:-1 * (pixel_cut[1])] * 255
            frame[(R_org_eye_roi_lt[0] + pixel_cut[0]):(R_org_eye_roi_lt[0] + R_org_eye_roi_size[0] - pixel_cut[0]),
            (R_org_eye_roi_lt[1] + pixel_cut[1]):(R_org_eye_roi_lt[1] + R_org_eye_roi_size[1] - pixel_cut[1])] = RE_infer[pixel_cut[0]:(-1 * pixel_cut[0]),
                                                                                                                 pixel_cut[1]:-1 * (pixel_cut[1])] * 255

        if draw_param:
            frame = self.draw_parameters(frame, alpha_w2c, self.Pe, R_w)

        return frame

    def correct_with_face_detection(self, frame, draw_param=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(gray, (self.face_detect_size[0], self.face_detect_size[1]))
        detections = self.detector(face_detect_gray, 0)
        frame = self.correct(frame, gray, detections, draw_param=draw_param)
        return frame


if __name__ == '__main__':

    import click


    @click.group()
    def cmd():
        pass


    @cmd.command()
    @click.option('--interpupillary_distance', '-i', type=float, default=6.3, help='your interpupillary distance.')
    @click.option('--dlib_shape_path', '-d', type=str, default=_DEFAULT_DLIB_DAT_PATH, help='path to dlib shape dat.')
    @click.option('--video_w', '-w', type=int, default=640, help='webcam frame width.')
    @click.option('--video_h', '-h', type=int, default=480, help='webcam frame height.')
    @click.option('--detect_w', '-dw', type=int, default=640, help='width of face detect size for dlib.')
    @click.option('--detect_h', '-dh', type=int, default=480, help='height of face detect size for dlib.')
    def focallength(interpupillary_distance, dlib_shape_path, video_w, video_h, detect_w, detect_h):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(dlib_shape_path)
        calibrator = FocalLengthCalibrator(interpupillary_distance, detector, predictor, (video_w, video_h), (detect_w, detect_h))
        calibrator.start()


    @cmd.command()
    @click.option('--dlib_shape_path', '-d', type=str, default=_DEFAULT_DLIB_DAT_PATH, help='path to dlib shape dat.')
    @click.option('--model_dir', '-m', type=str, default='resources/models/gaze_correction/weights/warping_model/flx/12', help='path to flx model parameter checkpoint dir')
    @click.option('--screen_w', '-sw', type=int, default=2560, help='screen width (pt).')
    @click.option('--screen_h', '-sh', type=int, default=1600, help='screen height (pt).')
    @click.option('--screen_w_cm', '-swcm', type=float, default=27.7, help='screen width (cm).')
    @click.option('--screen_h_cm', '-shcm', type=float, default=17.9, help='screen height (cm).')
    @click.option('--video_w', '-w', type=int, default=640, help='webcam frame width.')
    @click.option('--video_h', '-h', type=int, default=480, help='webcam frame height.')
    @click.option('--app_w', '-aw', type=int, default=640, help='width of application window.')
    @click.option('--app_h', '-ah', type=int, default=480, help='height of application widdow')
    @click.option('--camera_x', '-cx', type=float, default=0.0, help='camera x in global (cm).')
    @click.option('--camera_y', '-cy', type=float, default=-9.4, help='camera y in global (cm).')
    @click.option('--camera_z', '-cz', type=float, default=0.0, help='camera z in global (cm).')
    @click.option('--interpupillary_distance', '-i', type=float, default=6.3, help='your interpupillary distance.')
    @click.option('--focal_length', '-f', type=int, default=500, help='focal_length.')
    def correction(model_dir, dlib_shape_path, screen_w, screen_h, screen_w_cm, screen_h_cm, video_w, video_h, app_w, app_h, camera_x, camera_y, camera_z,
                   interpupillary_distance, focal_length):
        # 視線補正class
        r = GazeCorrector(
            dlib_dat_path=dlib_shape_path,
            model_dir=model_dir,
            screen_size_cm=(screen_w_cm, screen_h_cm),
            screen_size_pt=(screen_w, screen_h),
            app_window_rect=(screen_w / 2, screen_h / 2, app_w, app_h),
            video_size=(video_w, video_h),
            camera_pos_cm=(camera_x, camera_y, camera_z),
            interpupillary_distance_cm=interpupillary_distance,
            focal_length=focal_length
        )

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_h)

        window_name = "frame"
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, int(screen_w / 2), int(screen_h / 2))

        while True:

            ret, frame = cap.read()
            org_frame = frame.copy()
            # 視線補正関数
            frame = r.correct_with_face_detection(frame)

            cv2.imshow(window_name, frame)
            cv2.imshow("org", org_frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        pass


    cmd(obj={})