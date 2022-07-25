class args():
    # training args
    epochs = 50
    batch_size = 4
    in_chans = 1
    out_chans = 1
    embed_dim = 96
    window_size = 7
    dataset = "E:/COCO_224"

    save_model_dir = "D:/SwinFuse/models"  # "path to folder where trained model will be saved."
    save_loss_dir = "D:/SwinFuse/models/loss"  # "path to folder where trained model will be saved."

    height = 224
    width = 224
    image_size = 224  # "size of training images, default is 224 X 224"
    cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"
    seed = 42
    ssim_weight = [1, 10, 100, 1000, 10000]
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

    lr = 1e-5  # "learning rate"
    lr_light = 1e-5  # "learning rate"
    log_interval = 5  # "number of images after which the training loss is logged"
    log_iter = 1
    resume = None
    resume_auto_en = None
    resume_auto_de = None
    resume_auto_fn = None

    model_path_gray = "./model/Final_epoch_50_Mon_Feb_14_17_37_05_2022_1e3.model"
