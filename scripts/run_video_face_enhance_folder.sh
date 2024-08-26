input=/mnt/ec-data2/ivs/1080p/yongpeng/zhijian/sdr2hdr_v35_对比度过强/sdr片源
output=/mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/face_enhance
mkdir -p $output
for file in `ls $input`
do 
python ../inference/video_face_process.py --video_path $input/$file --save_path $output/${file%.*}_enh.mp4 --qp 12 --task enhance
# break
done