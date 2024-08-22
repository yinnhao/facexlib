# parsing
python ../inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_parsing.mp4 --task parsing --qp 20
# skin
python ../inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_skin.mp4 --task skin --qp 20
# enhance
python ../inference/video_face_process.py --video_path /mnt/ec-data2/ivs/1080p/zyh/hdr_dirty_face/sdr/jialin/SDR2822_709.mp4 --save_path /mnt/ec-data2/ivs/1080p/zyh/SDR2822_709_enhance.mp4 --task enhance --qp 12