import torch
import sys
import argparse
import time
import shutil
from utils import *
from config import TrainConfig
from inputs import train_arg_parser
import pprint
import pickle

class ModelConfig(TrainConfig):
    def __init__(self, inputs:dict) -> None:
        super().__init__()
        for param, value in inputs.items():
            setattr(self, param, value)

def get_data(
        model_config
):
    ## load data
    all_gt_img_data = []
    all_noisy_data = []
    for paths in model_config.data:
        all_gt_img_data, all_noisy_data = load_data(paths, 
                                                    all_gt_img_data, 
                                                    all_noisy_data, 
                                                    model_config.max_proj)

    ## prepare training data
    train_gt_img_data, train_noisy_data = prepare_training_data(all_gt_img_data, 
                                                                all_noisy_data, 
                                                                model_config.depth, 
                                                                model_config.mode)

    ## split data into patches
    # train_gt_img_data_patch, train_noisy_img_data_patch = to_patches(train_gt_img_data, train_noisy_data)
    train_gt_img_data_patch = train_gt_img_data
    train_noisy_img_data_patch = train_noisy_data


    ## subsample data based on tsize for training
    if model_config.tsize != 0:
        # if tsize is specified, use tsize number of images as train and rest as test
        split_ratio = (train_gt_img_data_patch.shape[0] - model_config.tsize)/train_gt_img_data_patch.shape[0]
        train_X, train_Y, test_X, test_Y = split_train_test(train_noisy_img_data_patch, train_gt_img_data_patch, split_ratio)
        tsize = train_X.shape[0]
    else:
        # if tsize is not specified, split full data for train and test
        train_X, train_Y, test_X, test_Y = split_train_test(train_noisy_img_data_patch, train_gt_img_data_patch, 0.33)
        tsize = train_X.shape[0]
    
    ## pytorch specific channel dimension permuting
    train_X, train_Y, test_X, test_Y = pytorch_specific_manipulations(train_X,
                                                                      train_Y,
                                                                      test_X,
                                                                      test_Y)
    
    return train_X, train_Y, test_X, test_Y, tsize

def get_model(
        model_config,
        in_channels
):
    model = get_cnn_arch_from_argin(model_config.arch)(in_channels=in_channels, out_channels=32)
    return model

def dummy_def():
    return 1

def train_step(
        model_config,
        epoch,
        tsize,
        train_X,
        train_Y,
        test_X, 
        test_Y,
        device,
        model,
        optimizer,
        loss_fn,
        file
):
    if tsize > 500:
        idx = random.sample(range(tsize), 500)
        curr_batch_X = train_X[idx, :, :, :]
        curr_batch_Y = train_Y[idx, :, :, :]
    else:
        curr_batch_X = train_X
        curr_batch_Y = train_Y
    tic = time.time()
    for batch in range(max(len(curr_batch_X) // model_config.bs, 1)):
        batch_x = curr_batch_X[batch * model_config.bs:min((batch + 1) * model_config.bs, len(curr_batch_X))]
        batch_y = curr_batch_Y[batch * model_config.bs:min((batch + 1) * model_config.bs, len(curr_batch_Y))]
        batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
        
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
    toc = time.time()

    # Calculate accuracy of 10 test images, repeat 5 times and report mean
    batch_test_loss = []
    for k in range(5):
        idx = [i for i in range(test_X.shape[0])]
        random.shuffle(idx)
        batch_x = test_X[idx[:min(10, test_X.shape[0])], :, :, :]
        batch_y = test_Y[idx[:min(10, test_Y.shape[0])], :, :, :]
        batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
        with torch.no_grad():
            pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        batch_test_loss.append(loss)

    # Calculate accuracy of 10 train images, repeat 5 times and report mean
    batch_train_loss = []
    for k in range(5):
        idx = [i for i in range(train_X.shape[0])]
        random.shuffle(idx)
        batch_x = train_X[idx[:min(10, train_X.shape[0])], :, :, :]
        batch_y = train_Y[idx[:min(10, train_Y.shape[0])], :, :, :]
        batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
        with torch.no_grad():
            pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        batch_train_loss.append(loss)

    mean_train_loss = sum(batch_train_loss) / len(batch_train_loss)
    mean_test_loss = sum(batch_test_loss) / len(batch_test_loss)
    print(f'epoch: {epoch}, Train Loss: {mean_train_loss}, Test Loss: {mean_test_loss}')
    file.write(f'{epoch},{mean_train_loss},{mean_test_loss},{toc-tic},{model_config.depth},{model_config.run},{model_config.tsize}\n')


def save_example_denoising_on_random_test_data(
        model_config,
        test_X,
        test_Y,
        model,
        device,
        results_dir
):
    print('saving denoising on random test examples')
    for i in range(10):
        temp_idx = random.randint(0, test_X.shape[0] - 1)
        batch_x = test_X[temp_idx, :, :, :]
        batch_y = test_Y[temp_idx, :, :, :]
        batch_x = batch_x[np.newaxis, :, :, :]
        batch_y = batch_y[np.newaxis, :, :, :]
        batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
        with torch.no_grad():
            pred = model(batch_x)
        batch_x = batch_x.cpu().numpy()
        batch_y = batch_y.cpu().numpy()
        pred = pred.cpu().numpy()
        save_name_X = os.path.join(results_dir, f'X_{temp_idx + 1}.png')
        save_name_Y = os.path.join(results_dir, f'Y_{temp_idx + 1}.png')
        save_name_pred = os.path.join(results_dir, f'pred_{temp_idx + 1}.png')
        if model_config.mode == '2D':
            cv2.imwrite(save_name_X, batch_x[0, 0, :, :].astype(np.uint16))  # this is the middle zplane corresponding to gt zplane
            cv2.imwrite(save_name_Y, batch_y[0, 0, :, :].astype(np.uint16))
            cv2.imwrite(save_name_pred, pred[0, 0, :, :].astype(np.uint16))
        elif model_config.mode == '2.5D':
            cv2.imwrite(save_name_X, batch_x[0, int((model_config.depth + 1) / 2 - 1), :, :].astype(np.uint16))  # this is the middle zplane corresponding to gt zplane
            cv2.imwrite(save_name_Y, batch_y[0, 0, :, :].astype(np.uint16))
            cv2.imwrite(save_name_pred, pred[0, 0, :, :].astype(np.uint16))
        elif model_config.mode == '3D':
            os.mkdir(os.path.join(results_dir, f'img_{temp_idx + 1}'))
            for z in range(pred.shape[3]):
                cv2.imwrite(os.path.join(results_dir, f'img_{temp_idx + 1}', f'X_z{z + 1}.png'), batch_x[0, z, :, :].astype(np.uint16))
                cv2.imwrite(os.path.join(results_dir, f'img_{temp_idx + 1}', f'Y_z{z + 1}.png'), batch_y[0, z, :, :].astype(np.uint16))
                cv2.imwrite(os.path.join(results_dir, f'img_{temp_idx + 1}', f'pred_z{z + 1}.png'), pred[0, z, :, :].astype(np.uint16))


def calculate_metrics(
        model_config,
        test_X,
        test_Y,
        model,
        loss_fn,
        device,
        results_dir
):
    file = open(os.path.join(results_dir, 'test_data_loss.txt'), 'a')
    idx = [i for i in range(test_X.shape[0])]
    random.shuffle(idx)
    for i in range(min(150, len(idx))):
        batch_x = test_X[idx[i], :, :, :]
        batch_y = test_Y[idx[i], :, :, :]
        batch_x = batch_x[np.newaxis, :, :, :]
        batch_y = batch_y[np.newaxis, :, :, :]
        batch_x = torch.tensor(batch_x, dtype=torch.float).to(device)
        batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
        tic = time.time()
        with torch.no_grad():
            pred = model(batch_x)
        toc = time.time()
        loss = loss_fn(pred, batch_y)
        file.write(f'{i},{idx[i] + 1},{loss},{toc-tic},{model_config.depth},{model_config.run},{model_config.tsize}\n')
    file.close()


def trainer(model_config):
    # collate data
    train_X, train_Y, test_X, test_Y, tsize = get_data(model_config)

    ## define CNN model
    model = get_model(model_config, in_channels=train_X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def loss_fn(pred, y):
        if model_config.loss == 'l2':
            loss = torch.mean((y - pred)**2)
        elif model_config.loss == 'l1':
            loss = torch.mean(torch.abs(y - pred))
        return loss

    ## make folder where all results will be saved
    base_save_path = f'run_{model_config.arch}_'\
                    f'{model_config.loss}_'\
                    f'mp{model_config.max_proj}_'\
                    f'm{model_config.mode}_'\
                    f'd{model_config.depth}_'\
                    f'{model_config.run}_'\
                    f'{model_config.tsize}'
    if model_config.out != '':
        if os.path.isdir(model_config.out) == False:
            os.mkdir(model_config.out)
        results_dir = os.path.join(model_config.out, base_save_path)
    else:
        results_dir = base_save_path
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    ## start training
    pprint.pprint('Starting training')
    pprint.pprint(vars(model_config))
    file = open(os.path.join(results_dir, 'training_loss.txt'), 'a')
    for epoch in range(model_config.epochs):
        train_step(
            model_config,
            epoch,
            tsize,
            train_X,
            train_Y,
            test_X,
            test_Y,
            device,
            model,
            optimizer,
            loss_fn,
            file
        )
    file.close()

    # save model config and model wieghts
    model_config.in_channels = model.in_channels
    model_config.out_channels = model.out_channels
    model_config.tsize = tsize
    save_config_path = os.path.join(results_dir, 'model_config.pickle')
    save_model_path = os.path.join(results_dir, 'model_weights.pt')
    with open(save_config_path, 'wb') as handle:
        pickle.dump(vars(model_config), handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'saved model config at {save_config_path}')
    torch.save(model.state_dict(), save_model_path)
    print(f'saved model weights at {save_model_path}')

    # save some random prediction examples
    save_example_denoising_on_random_test_data(
        model_config,
        test_X,
        test_Y,
        model,
        device,
        results_dir
    )

    # calculate accuracy on test data and save results
    calculate_metrics(
        model_config,
        test_X,
        test_Y,
        model,
        loss_fn,
        device,
        results_dir
    )


if __name__ == '__main__':
    
    model_config = ModelConfig(train_arg_parser())
    trainer(model_config)