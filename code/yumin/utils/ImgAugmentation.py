import Augmentor

train256_path = '/home/ubuntu/dl4cvproject/data/train256'
p=Augmentor.Pipeline(train256_path)
p.rotate90(probability=0.3)
p.rotate270(probability=0.3)
p.flip_left_right(probability=0.3)
p.flip_top_bottom(probability=0.3)
p.sample(1000)
