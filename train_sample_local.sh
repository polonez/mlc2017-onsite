python gan/train.py --train_data_pattern='gan/train.tfrecords' \
--generator_model=SampleGenerator --discriminator_model=SampleDiscriminator \
--train_dir=/tmp/kmlc_gan_train --num_epochs=500 --export_generated_images --start_new_model
