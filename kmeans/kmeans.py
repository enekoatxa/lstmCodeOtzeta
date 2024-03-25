import torch
import numpy as np
from kmeans_pytorch import kmeans, kmeans_predict
import sys
sys.path.insert(1, '/home/bee/Desktop/idle animation generator/code/util')
import bvhLoader


# set device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# data
data_size, dims, num_clusters = 498, 1, 2000
x = bvhLoader.loadDatasetForVae("silenceDataset3sec", partition="Train", specificSize=800000, verbose=True)
all_clusters=[]
for a in range(0, len(x), 2):
    all_clusters.append(x[a])
all_clusters = np.asarray(all_clusters)
all_clusters = torch.from_numpy(all_clusters)
print(len(all_clusters))
# print(len(x))
# x1 = np.asarray(x[0:5000])
# x1 = torch.from_numpy(x1)

# x2 = np.asarray(x[5001:10000])
# x2 = torch.from_numpy(x2)

# x3 = np.asarray(x[10001:15000])
# x3 = torch.from_numpy(x3)

# x4 = np.asarray(x[15001:20000])
# x4 = torch.from_numpy(x4)

# x5 = np.asarray(x[20001:25000])
# x5 = torch.from_numpy(x5)

# x6 = np.asarray(x[25001:30000])
# x6 = torch.from_numpy(x6)

# x7 = np.asarray(x[30001:35000])
# x7 = torch.from_numpy(x7)

# x8 = np.asarray(x[35001:40000])
# x8 = torch.from_numpy(x8)

# x9 = np.asarray(x[40001:45000])
# x9 = torch.from_numpy(x9)

# x10 = np.asarray(x[45001:50000])
# x10 = torch.from_numpy(x10)

# # kmeans
# cluster_ids_x1, cluster_centers1 = kmeans(
#     X=x1, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x2, cluster_centers2 = kmeans(
#     X=x2, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x3, cluster_centers3 = kmeans(
#     X=x3, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x4, cluster_centers4 = kmeans(
#     X=x4, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x5, cluster_centers5 = kmeans(
#     X=x5, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x6, cluster_centers6 = kmeans(
#     X=x6, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x7, cluster_centers7 = kmeans(
#     X=x7, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x8, cluster_centers8 = kmeans(
#     X=x8, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x9, cluster_centers9 = kmeans(
#     X=x9, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# # kmeans
# cluster_ids_x10, cluster_centers10 = kmeans(
#     X=x10, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0'), tol=0.0000000000000000000000000000000001
# )

# all_clusters = torch.cat((cluster_centers1, cluster_centers2, cluster_centers3, cluster_centers4, cluster_centers5,
#                           cluster_centers6, cluster_centers7, cluster_centers8, cluster_centers9, cluster_centers10), 0)
print(len(all_clusters))
header, sample = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/testBvh.bvh", returnHeader=True)
file = open("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/cosineDecodedTestBvh.bvh", "w")
file.write(header)
sample = np.asarray(sample)
sample = torch.from_numpy(sample)
# newFrameIndexes = kmeans_predict(X=sample, cluster_centers=all_clusters, distance='euclidean', device=device)
# print(newFrameIndexes)
for index in sample:
    print("index")
    bestOutput = 999999999.0
    for target in all_clusters:
        dist = torch.nn.CosineSimilarity(dim=0)
        output = dist(target, index)
        if(output.item()<bestOutput):
            bestOutput = output.item()
            bestTarget = torch.clone(target)
    file.write(str(bestTarget.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
    file.write("\n")
file.close()

header, sample = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/testBvh.bvh", returnHeader=True)
file = open("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/cosineDecodedTestBvh.bvh", "w")
file.write(header)
sample = np.asarray(sample)
sample = torch.from_numpy(sample)
# newFrameIndexes = kmeans_predict(X=sample, cluster_centers=all_clusters, distance='euclidean', device=device)
# print(newFrameIndexes)
for index in sample:
    print("index")
    bestOutput = 999999999.0
    for target in all_clusters:
        dist = torch.nn.CosineSimilarity(dim=0)
        output = dist(target, index)
        if(output.item()<bestOutput):
            bestOutput = output.item()
            bestTarget = torch.clone(target)
    file.write(str(bestTarget.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
    file.write("\n")
file.close()

header, sample = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/testBvh2.bvh", returnHeader=True)
file = open("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/cosineDecodedTestBvh2.bvh", "w")
file.write(header)
sample = np.asarray(sample)
sample = torch.from_numpy(sample)
# newFrameIndexes = kmeans_predict(X=sample, cluster_centers=all_clusters, distance='euclidean', device=device)
# print(newFrameIndexes)
for index in sample:
    print("index")
    bestOutput = 999999999.0
    for target in all_clusters:
        dist = torch.nn.CosineSimilarity(dim=0)
        output = dist(target, index)
        if(output.item()<bestOutput):
            bestOutput = output.item()
            bestTarget = torch.clone(target)
    file.write(str(bestTarget.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
    file.write("\n")
file.close()

header, sample = bvhLoader.loadBvhToList("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/testBvh3.bvh", returnHeader=True)
file = open("/home/bee/Desktop/idle animation generator/code/geneaDatasetCreation/cosineDecodedTestBvh3.bvh", "w")
file.write(header)
sample = np.asarray(sample)
sample = torch.from_numpy(sample)
# newFrameIndexes = kmeans_predict(X=sample, cluster_centers=all_clusters, distance='euclidean', device=device)
# print(newFrameIndexes)
for index in sample:
    print("index")
    bestOutput = 999999999.0
    for target in all_clusters:
        dist = torch.nn.CosineSimilarity(dim=0)
        output = dist(target, index)
        if(output.item()<bestOutput):
            bestOutput = output.item()
            bestTarget = torch.clone(target)
    file.write(str(bestTarget.cpu().numpy().tolist()).replace("[", "").replace("]", "").replace(",", ""))
    file.write("\n")
file.close()
# y = bvhLoader.loadDatasetForVae("silenceDataset3sec", partition="Validation", specificSize=300)
# y = np.asarray(y)
# y = torch.from_numpy(y)
# cluster_ids_y = kmeans_predict(
#     y, cluster_centers, 'euclidean', device=device
# )
# print(cluster_ids_y)