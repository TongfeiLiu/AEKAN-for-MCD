from sklearn import metrics
import numpy as np
import cv2


class Evaluation():
    def __init__(self, label, pred):
        super(Evaluation, self).__init__()
        if label.max() > 1 or pred.max() > 1:
            self.label = label / 255.0
            self.pred = pred / 255.0
        else:
            self.label = label
            self.pred = pred
        if self.label.shape != self.pred.shape:
            raise ValueError("The shape of the label and prediction must be the same.")

        size = self.label.size

        # 计算交集和并集
        intersection = (self.label * self.pred)  # 交集 (TP)
        union = np.clip(self.label + self.pred, 0, 1)  # 并集 (TP + FP + FN)

        # 计算 TP, TN, FP, FN
        self.TP = int(intersection.sum())  # True Positive
        self.TN = int(((1 - self.label) * (1 - self.pred)).sum())  # True Negative
        self.FP = int(((self.pred == 1) & (self.label == 0)).sum())  # False Positive
        self.FN = int(((self.label == 1) & (self.pred == 0)).sum())  # False Negative

        self.c_num_or = int(union.sum())  #  (TP + FP + FN)
        self.uc_num_or = int(size - self.c_num_or)  #  (TN)

    def matrix(self):
        """ Returns the confusion matrix(TP, TN, FP, FN) """
        return self.TP, self.TN, self.FP, self.FN

    def Classification_indicators(self):
        """ Calculating classification metrics（OA, Kappa, AA） """
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        OA = (TP + TN) / (TP + TN + FP + FN)


        kappa = metrics.cohen_kappa_score(self.label.flatten(), self.pred.flatten())


        AA = (TP / (TP + FN) + TN / (TN + FP)) / 2

        return OA * 100, kappa * 100, AA * 100

    def CD_indicators(self):
        """ Calculating change detection metrics（FA, MA, TE）"""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        # 假警报率 (FA)
        FA = FP / (FP + TN) if (FP + TN) != 0 else 0

        # 漏检率 (MA)
        MA = FN / (FN + TP) if (FN + TP) != 0 else 0

        # 总误差率 (TE)
        TE = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

        return FA * 100, MA * 100, TE * 100

    def Landslide_indicators(self):
        """ （Completeness, Correctness, Quality）"""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN


        Completeness = TP / (TP + FN) if (TP + FN) != 0 else 0


        Correctness = TP / (TP + FP) if (TP + FP) != 0 else 0


        Quality = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

        return Completeness * 100, Correctness * 100, Quality * 100

    def IOU_indicator(self):
        """ 计算IoU指标（mIoU, c_iou, uc_iou）"""
        c_num_and = self.TP  # Change area intersection
        c_num_or = self.c_num_or  # Change Region Union
        uc_num_and = self.TN  # Unchanged Area Intersection
        uc_num_or = self.uc_num_or  # Unchanged Region Union

        # Change area IoU
        c_iou = (c_num_and / c_num_or) * 100 if c_num_or != 0 else 0

        # IoU of unchanged area
        uc_iou = (uc_num_and / uc_num_or) * 100 if uc_num_or != 0 else 0

        # Average IoU
        mIoU = (c_iou + uc_iou) / 2

        return mIoU, c_iou, uc_iou

    def ObjectExtract_indicators(self):
        """ Calculate target extraction metrics（Precision, Recall, F1）"""
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        # Precision
        Precision = TP / (TP + FP) if (TP + FP) != 0 else 0


        Recall = TP / (TP + FN) if (TP + FN) != 0 else 0


        epsilon = 1e-7  # Avoid division by zero
        F1 = (2 * Precision * Recall) / (Precision + Recall + epsilon) if (Precision + Recall) != 0 else 0

        return Precision * 100, Recall * 100, F1 * 100


if __name__ == "__main__":
    pred_path = "/opt/data/private/xj/most_new_experience1/CM_iter_20.png"
    label_path = "/opt/data/private/xj/most_new_experience1/data/dataset/Img7-C.png"
    pred = cv2.imread(pred_path)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    Indicators = Evaluation(label, pred)
    OA, kappa, AA = Indicators.Classification_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    CP, CR, AQ = Indicators.Landslide_indicators()
    IOU = Indicators.IOU_indicator()
    Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    print("(OA, KC, AA)", OA, kappa, AA)
    print("(FA, MA, TE)", FA, MA, TE)
    print("(CP, CR, AQ)", CP, CR, AQ)
    print("(IoU, Precision, Recall, F1-score)", IOU, Precision, Recall, F1)

