from PIL import Image
import torchvision.transforms as transforms
import torch.optim

def show_demo(reshape_size, data_root):
    trans = transforms.Compose([
        transforms.Resize(reshape_size),
        transforms.ToTensor(),
    ]
    )

    img = os.path.join(data_root, os.listdir(data_root)[0])
    demo = Image.open(img)
    demo_img = trans(demo)
    # np.moveaxis : move the dimension to the new position
    demo_array = np.moveaxis(demo_img.numpy() * 255, 0, -1)

    return Image.fromarray(demo_array.astype(np.uint8))


def get_optimizer(args, D, G):

    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    return optim_D, optim_G