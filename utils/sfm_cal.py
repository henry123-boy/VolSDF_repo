import pdb

from . import sfm_utils
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
class SfMer():
    def __init__(self,opt,intrinsics,imgs):
        self.opt=opt
        self.intrinsics=intrinsics
        self.imgs=imgs
        self.triangles=[]
        #self.Delaunay_Tri(self.pairs[0])
        #self.Delaunay_Tri(self.pairs[1])
    def Delaunay_Tri(self,points):
        tri = Delaunay(points)
        # center = np.sum(points[tri.simplices], axis=1) / 3.0
        # color = []
        # for index, sim in enumerate(points[tri.simplices]):
        #     cx, cy = center[index][0], center[index][1]
        #     x1, y1 = sim[0][0], sim[0][1]
        #     x2, y2 = sim[1][0], sim[1][1]
        #     x3, y3 = sim[2][0], sim[2][1]
        #
        #     s = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5 + ((cx - x3) ** 2 + (cy - y3) ** 2) ** 0.5 + (
        #                 (cx - x2) ** 2 + (cy - y2) ** 2) ** 0.5
        #     color.append(s)
        # color = np.array(color)
        # plt.figure(figsize=(20, 10))
        # plt.tripcolor(points[:, 0], points[:, 1], tri.simplices.copy(), facecolors=color, edgecolors='k')
        #
        # plt.tick_params(labelbottom='off', labelleft='off', left='off', right='off', bottom='off', top='off')
        # ax = plt.gca()
        # plt.scatter(points[:, 0], points[:, 1], color='r')
        # # plt.grid()
        # plt.savefig('Delaunay.png', transparent=True, dpi=600)
        self.triangles.append(tri.simplices)
        # self.vis_tri(self.imgs[0], self.pairs[0], self.triangles[0], 0)
        # self.vis_tri(self.imgs[1], self.pairs[1], self.triangles[0], 1)
    def vis_tri(self,img,points,tri,i):
        for index, sim in enumerate(points[tri]):
            x1, y1 = sim[0][0], sim[0][1]
            x2, y2 = sim[1][0], sim[1][1]
            x3, y3 = sim[2][0], sim[2][1]
            pt1=(int(x1),int(y1))
            pt2 = (int(x2), int(y2))
            pt3 = (int(x3), int(y3))
            cv2.line(img,pt1,pt2,thickness=2,color=(255,255,255))
            cv2.line(img, pt2, pt3, thickness=2, color=(255, 255, 255))
            cv2.line(img, pt3, pt1, thickness=2, color=(255, 255, 255))
        cv2.imwrite("./tri_{}.jpg".format(i),img)

    def cal_pose_pts3D(self):
        self.pairs = sfm_utils.sift_matches(self.imgs[0], self.imgs[1])
        sfm_utils.recover_pose(self, self.pairs[0], self.pairs[1], self.intrinsics)
        self.pts3D = sfm_utils.trangle_3Dpts(self.pose1, self.pose2, self.intrinsics, self.intrinsics,
                                             self.pairs[0], self.pairs[1])
        print("the filtered points number is {}".format(self.pts3D.shape[-1]))


# opt=None
# img1_path=r"../datasets/ETH3D/DSC_0259.JPG"
# img2_path=r"../datasets/ETH3D/DSC_0260.JPG"
# #3410.68 3409.64 3115.69 2063.73
# K=np.array([[3410.68/8,0,3115.69/8],[0,3409.64/8,2063.73/8],[0,0,1]])
# img1=cv2.imread(img1_path)
# img2=cv2.imread(img2_path)
# x, y = img1.shape[0:2]
# img1=cv2.resize(img1,(int(y/8),int(x/8)))
# img2=cv2.resize(img2,(int(y/8),int(x/8)))
# sfmer=SfMer(opt,K,[img1,img2])
# SfMer.cal_pose_pts3D()
# SfMer.Delaunay_Tri(sfmer.pairs[0])
