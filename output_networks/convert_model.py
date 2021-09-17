import os
import paddle
import torch


def save_to_dir(saved_path, dir):
    if dir:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f"create new dir: {dir}")
        return os.path.join(dir, saved_path)
    return saved_path


def convert_model(model_path, saved_dir=None):
    tm = torch.load(model_path, map_location='cpu')

    paddle_state_dict = {'config': tm['config'], 'netG': {}, 'netD': {}, 'avgG': {}}

    # 需要转置：
    # tran_lst = ["formatLayer.module.weight", "groupScaleZero.1.module.weight", "decisionLayer.module.weight"]

    for key in list(tm.keys())[1:]:  # 跳过 'config'
        print("key:", key)
        for name, val in tm[key].items():
            if len(val.shape) == 2:  # 全连接层需要转置
                print(f"{name}'s val.shape: {val.shape} needs trans")
                val = val.t()
                print(f"to val.shape: {val.shape}")
            paddle_state_dict[key][name] = val.detach().numpy()

    saved_path = model_path.split('/')[-1].replace('.pth', '.pdparams')
    saved_path = save_to_dir(saved_path, dir=saved_dir)

    paddle.save(paddle_state_dict, path=saved_path)
    print(f"convert finished! model saved to {saved_path}")


def convert_refVector(vector_path, saved_dir=None):
    tv = torch.load(vector_path, map_location='cpu')

    saved_path = vector_path.split('/')[-1].replace('.pt', '.pdparams')
    saved_path = save_to_dir(saved_path, dir=saved_dir)

    paddle.save(tv.detach().numpy(), path=saved_path)
    print(f"convert finished! refVector saved to {saved_path}")


if __name__ == '__main__':
    model_path = "./ckpt_snap/official/celeba_cropped_s5_i83000.pth"
    # vector_path = "./ckpt_snap/s5+/celeba_cropped_refVectors.pt"
    to_dir = "./ckpt_snap/best_swd"

    # convert_refVector(vector_path, saved_dir=to_dir)
    convert_model(model_path, saved_dir=to_dir)
