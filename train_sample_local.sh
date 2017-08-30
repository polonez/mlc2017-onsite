python gan/train.py --train_data_pattern='gan/train.tfrecords' \
--generator_model=SmallerGenerator --discriminator_model=SmallerDiscriminator \
--train_dir=/tmp/kmlc_gan_train --num_epochs=500 \
--start_new_model
#--image_dir='smaller-kernel/' --export_generated_images
