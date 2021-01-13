batch_size="500"
kernel_sizes="3,4,5"
dim="300"
kernel_num="100"
learning_rate="0.001"
epochs="100"

python3 main.py \
--batch-size=${batch_size} \
--kernel-sizes=${kernel_sizes} \
--dim=${dim} \
--kernel-num=${kernel_num} \
--lr=${learning_rate} \
--epochs=${epochs} \
--train