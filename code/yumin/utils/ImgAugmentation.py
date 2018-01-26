import Augmentor

prefixTr256 = '/home/ubuntu/dl4cvproject/data/play'
p=Augmentor.Pipeline(prefixTr256)
p.rotate90(probability=0.3)
p.rotate270(probability=0.3)
p.flip_left_right(probability=0.3)
p.flip_top_bottom(probability=0.3)
p.sample(1000)
