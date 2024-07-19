import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchinfo import summary

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Old weights with accuracy 76.130%
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    info = summary(model, (1, 3, 224, 224), verbose=0, col_names=("output_size", "num_params", "mult_adds"))
    print(info.max_memory, info.total_mult_adds, info.memory_layer)
