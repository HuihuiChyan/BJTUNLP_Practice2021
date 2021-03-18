batch_size="128"
embedding_dim="300"
hidden_dim="200"
learning_rate="0.0002"
epoches="50"

# nohup python3 main.py \
# --batch-size=${batch_size} \
# --embedding-dim=${embedding_dim} \
# --hidden-dim=${hidden_dim} \
# --learning-rate=${learning_rate} \
# --epoches=${epoches} \
# --train --cuda &

python3 main.py \
--batch-size=${batch_size} \
--embedding-dim=${embedding_dim} \
--hidden-dim=${hidden_dim} \
--learning-rate=${learning_rate} \
--epoches=${epoches} \
