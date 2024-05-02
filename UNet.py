from package.model import *
from package.function import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


modeltype = 'UNet'

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

for fold in range(4,6):
    # Training data generator
    trainGene = trainGenerator(batch_size=10,
                            train_path=f'ETT_v3/Fold{fold}',
                            image_folder='train',
                            mask_folder='trainannot',
                            aug_dict=data_gen_args,
                            save_to_dir=None)

    # Validation data generator
    valGene = trainGenerator(batch_size=10,
                            train_path=f'ETT_v3/Fold{fold}',
                            image_folder='val',
                            mask_folder='valannot',
                            aug_dict=data_gen_args,
                            save_to_dir=None)
    
    # Model creation
    model = unet()
    model_checkpoint = ModelCheckpoint(f"best_model/{modeltype}/Fold{fold}.keras", monitor='val_loss', verbose=0, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True)

    # Model training
    model.fit(trainGene,
            steps_per_epoch=128,  # Adjust based on the size of your dataset
            epochs=64,  # Total number of epochs to run
            callbacks=[model_checkpoint, early_stopping],
            verbose=1,
            validation_data=valGene,
            validation_steps=64)

               