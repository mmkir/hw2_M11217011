import os
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as trans
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,ReduceLROnPlateau
from skimage import img_as_ubyte
from tensorflow.keras.optimizers import Adam

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def count_files(directory):
    """計算指定資料夾內的檔案數量"""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


def save_images(img, mask, directory,num):
    # 建立目錄如果它不存在
    os.makedirs(directory, exist_ok=True)
    # 保存影像
    plt.imsave(os.path.join(directory, f"{num}.png"), img.squeeze(), cmap='gray')
    # 保存mask
    plt.imsave(os.path.join(directory, f"{num}_truth.png"), mask.squeeze(), cmap='gray')

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path, 
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path, num_image=30, target_size=(256,256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, f"{i}.png"), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if not flag_multi_class else img
        img = np.reshape(img, (1,) + img.shape)
        yield (img,)

def test_aug_generator(fold):
    test_folder = os.path.join(f'ETT_aug/Fold{fold}', 'test')

    for image_filename in os.listdir(test_folder):
        if image_filename.endswith('_predict.png'):
            pass
        elif image_filename.endswith('_truth.png'):
            pass
        else:
            image_path = os.path.join(test_folder, image_filename)
            image = Image.open(image_path)
            image = image.convert('L') 
            image_np = np.array(image) / 255.0
            image_np = np.expand_dims(image_np, axis=-1)
            image_np = np.expand_dims(image_np, axis=0)
            yield (image_np,)

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class, COLOR_DICT, item)
        else:
            img = item[:, :, 0]  # 假設這是浮點數據

        # 將圖像數據正規化到 0-1 範圍
        img = np.clip(img, 0, 1)

        # 將浮點數據轉換為 uint8
        img = img_as_ubyte(img)

        # 保存圖像
        io.imsave(os.path.join(save_path, f"{i}_predict.png"), img)

def create_comparison_image(fold,modeltype):
    base_path = f"ETT_aug/Fold{fold}/test"
    predict_path = f"ETT_predict/{modeltype}/Fold{fold}"

    image_files = count_files(predict_path)
    
    for img_name in range(image_files):
        img_path = os.path.join(base_path, f"{img_name}.png")
        gt_path = os.path.join(base_path, f"{img_name}_truth.png")
        pred_path = os.path.join(predict_path, f"{img_name}_predict.png")
        
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        pred = Image.open(pred_path)
        
        # 使用 matplotlib 來展示和儲存圖片
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # 設定一行三列
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[0].axis('off')  # 不顯示坐標軸

        axs[1].imshow(gt)
        axs[1].set_title('Ground Truth Mask')
        axs[1].axis('off')

        axs[2].imshow(pred)
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')

        output_path = os.path.join(f"ETT_compared/{modeltype}/Fold{fold}", f"{img_name}.png")
        plt.savefig(output_path)  # 儲存整個圖表
        plt.close(fig)  # 關閉圖形視窗以釋放資源
        print(f"Saved: {output_path}")



