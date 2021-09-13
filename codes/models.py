# -*- encoding: utf-8 -*-
"""
@Author : BIGBOSS_FoX
@File   : models.py
@Tel    : 13817043340
@Email  : chendymaodai@163.com
@Time   : 2021/9/7 下午2:09
@Desc   : Define PointNet, VFE and VFE_LW model architectures
"""
import torch
import torch.nn as nn


class VFE(nn.Module):
    def __init__(self, point_nums=1000):
        super(VFE, self).__init__()
        self.point_nums = point_nums
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.conv4 = nn.Conv1d(2048, 2048, 1)
        self.conv5 = nn.Conv1d(2048, 2048, 1)
        self.conv6 = nn.Conv1d(2048, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(2048)
        self.bn5 = nn.BatchNorm1d(2048)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # [N, 3, 1000]
        x = self.relu(self.bn1(self.conv1(x)))
        # [N, 64, 1000]
        x = self.relu(self.bn2(self.conv2(x)))
        # [N, 128, 1000]
        pwf1 = self.bn3(self.conv3(x))
        # 1st point-wise feature: [N, 1024, 1000]
        gf1 = torch.max(pwf1, 2, keepdim=True)[0]
        # 1st global feature: [N, 1024, 1]
        gaf1 = gf1.repeat(1, 1, self.point_nums)
        # 1st globally aggregated feature: [N, 1024, 1000]
        pwcf = torch.cat((pwf1, gaf1), dim=1)
        # point-wise concat feature: [N, 2048, 1000]
        pwcf = self.relu(self.bn4(self.conv4(pwcf)))
        # [N, 2048, 1000]
        pwcf = self.relu(self.bn5(self.conv5(pwcf)))
        # [N, 2048, 1000]
        pwcf = self.bn6(self.conv6(pwcf))
        # [N, 1024, 1000]
        gf2 = torch.max(pwcf, 2, keepdim=True)[0]
        # 2nd global feature: [N, 1024, 1]
        gf = gf2.view(-1, 1024)
        # global feature vector: [N, 1024]
        out = self.relu(self.bn7(self.fc1(gf)))
        # [N, 512]
        out = self.relu(self.bn8(self.dropout(self.fc2(out))))
        # [N, 256]
        out = self.fc3(out)
        # [N, 4]
        return out


class VFE_LW(nn.Module):
    """Light-weight version of VFE"""
    def __init__(self, point_nums=1000):
        super(VFE_LW, self).__init__()
        self.point_nums = point_nums
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.conv5 = nn.Conv1d(1024, 1024, 1)
        self.conv6 = nn.Conv1d(1024, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # [N, 3, 1000]
        x = self.relu(self.bn1(self.conv1(x)))
        # [N, 64, 1000]
        x = self.relu(self.bn2(self.conv2(x)))
        # [N, 128, 1000]
        pwf1 = self.bn3(self.conv3(x))
        # 1st point-wise feature: [N, 256, 1000]
        gf1 = torch.max(pwf1, 2, keepdim=True)[0]
        # 1st global feature: [N, 256, 1]
        gaf1 = gf1.repeat(1, 1, self.point_nums)
        # 1st globally aggregated feature: [N, 256, 1000]
        pwcf = torch.cat((pwf1, gaf1), dim=1)
        # point-wise concat feature: [N, 512, 1000]
        pwcf = self.relu(self.bn4(self.conv4(pwcf)))
        # [N, 1024, 1000]
        pwcf = self.relu(self.bn5(self.conv5(pwcf)))
        # [N, 1024, 1000]
        pwcf = self.bn6(self.conv6(pwcf))
        # [N, 1024, 1000]
        gf2 = torch.max(pwcf, 2, keepdim=True)[0]
        # 2nd global feature: [N, 1024, 1]
        gf = gf2.view(-1, 1024)
        # global feature vector: [N, 1024]
        out = self.relu(self.bn7(self.dropout(self.fc1(gf))))
        # [N, 512]
        out = self.relu(self.bn8(self.dropout(self.fc2(out))))
        # [N, 256]
        out = self.fc3(out)
        # [N, 4]
        return out


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # [N, 3, 1000]
        x = self.relu(self.bn1(self.conv1(x)))
        # [N, 64, 1000]
        x = self.relu(self.bn2(self.conv2(x)))
        # [N, 128, 1000]
        x = self.bn3(self.conv3(x))
        # [N, 1024, 1000]
        x = torch.max(x, 2, keepdim=True)[0]
        # [N, 1024, 1]
        x = x.view(-1, 1024)
        # [N, 1024]
        x = self.relu(self.bn4(self.fc1(x)))
        # [N, 512]
        x = self.relu(self.bn5(self.dropout(self.fc2(x))))
        # [N, 256]
        x = self.fc3(x)
        # [N, 4]
        return x


if __name__ == '__main__':
    pn = PointNet()
    vfe = VFE(point_nums=1000)
    vfe_lw = VFE_LW(point_nums=1000)
    sim_data = torch.rand(5, 3, 1000)
    print('Input tensor: ', sim_data.size())
    pn_out = pn(sim_data)
    print('PointNet: ', pn_out.size())
    vfe_out = vfe(sim_data)
    print('VFE: ', vfe_out.size())
    vfe_lw_out = vfe_lw(sim_data)
    print('VFE_LW: ', vfe_lw_out.size())
