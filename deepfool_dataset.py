import numpy as np
from deepfool import deepfool

def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v



def deepfool_dataset(dataset, f, grads, num_classes=10, overshoot=0.02, max_iter=100):
    num_images =  int(np.shape(dataset)[0])
    deepfool_datas=np.zeros(dataset.shape)
    for x in range(num_images):
        _ , _ ,label, fool_label , deepfool_data_temp=deepfool(dataset[x:x+1],f,grads,num_classes=num_classes,overshoot=overshoot,max_iter=max_iter)
        deepfool_datas[x]=deepfool_data_temp[0]
        print(str(x)+"th progress : "+str(label)+" ---> "+str(fool_label))

    fooling_rate=fooling_rate_calc_datasets(dataset_original=dataset,dataset_perts=deepfool_datas,f=f,batch_size=100)

    return deepfool_datas,fooling_rate
