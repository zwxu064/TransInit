import torch
import torch.nn.functional as F

# a_v = torch.FloatTensor([10, 20])
# a_v.requires_grad = True
# a_loc = torch.LongTensor([[1, 2],
#                           [5, 7]])
# print(a_loc.shape, a_v.shape)
# a = torch.sparse.FloatTensor(a_loc, a_v, torch.Size([3000, 512*512])).cuda()
# b = torch.ones((512*512, 21), dtype=torch.float32, requires_grad=False).cuda()
# mmul_sparse = torch.sparse.mm(a, b)
# loss = mmul_sparse.sum()
# loss.backward()
# print(a_v.grad)

#####
img_h, img_w, num_classes, batch = 512, 500, 21, 8
num_spixels = 1024
num_pixels = img_h * img_w
a_v = torch.ones(num_pixels, dtype=torch.float32)
a_loc_idx = torch.arange(0, num_pixels, dtype=torch.float32).unsqueeze(0)
scores = torch.randn((batch, num_classes, num_pixels), dtype=torch.float32, requires_grad=True).cuda()
sp_scores = []

for batch_idx in range(batch):
  a_loc_sp = torch.randint(0, num_spixels, (1, num_pixels), dtype=torch.float32)
  a_loc_sp[a_loc_sp == 0] = 2
  a_loc_sp[a_loc_sp == 1] = 2
  a_loc_sp[:, 0:4] = 0
  a_loc_sp[:, 4:8] = 1

  a_loc = torch.cat((a_loc_sp, a_loc_idx), dim=0).long()  # 2*(img_h*img_w) xy
  a = torch.sparse.FloatTensor(a_loc, a_v, torch.Size([num_spixels, num_pixels])).cuda()
  sp_nums = torch.sparse.sum(a, dim=1).unsqueeze(1)  # number of pixels for each sp
  score = scores[batch_idx].transpose(0, 1)
  print(batch_idx, score[0:4, :].mean(0).sum(), score[4:8, :].mean(0).sum())
  sp_score = torch.sparse.mm(a, score) / sp_nums.to_dense()  # num_spixels*num_classes

  score_new = torch.sparse.mm(a.transpose(0, 1), sp_score)  # (img_h*img_w)*num_classes
  score_new = score_new.permute(1, 0).view(-1, img_h, img_w)  # num_classes*img_h*img_w
  print(batch_idx, '===>', score_new[:, 0, 0].sum(), score_new[:, 0, 6].sum())
  sp_scores.append(score_new)

sp_scores = torch.stack(sp_scores, dim=0)  # batch*num_classes*img_h*img_w

loss = sp_scores.sum()
loss.backward()