import torch
from generator import Generator
from discriminator import Discriminator
low_resolution = 24 # 96x96 -> 24x24
with torch.cuda.amp.autocast():
    x = torch.randn((5, 3, low_resolution, low_resolution))
    gen = Generator()
    gen_out = gen(x)
    disc = Discriminator()
    disc_out = disc(gen_out)

    print(gen_out.shape)
    print(disc_out.shape)
