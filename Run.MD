Three files(DataProcessing.py, model_main_tf2.py, exporter_main_v2.py) are command line scripts.  

Steps for Training:-  
1. Exract *.zip in images folder
2. `python DataProcessing.py -r 0.5`
3. `python model_main_tf2.py --model_dir=models/ssd_resnet152_v1_fpn --pipeline_config_path=models/ssd_resnet152_v1_fpn/pipeline.config` 

Evaluation:-   
`python model_main_tf2.py --model_dir=/home/tushar/Desktop/EagleView/TensorFlow/workspace/ODv1/exported-models/my_model/saved_model --pipeline_config_path=/home/tushar/Desktop/EagleView/TensorFlow/workspace/ODv1/exported-models/my_model/pipeline.config --checkpoint_dir=/home/tushar/Desktop/EagleView/TensorFlow/workspace/ODv1/exported-models/my_model/checkpoint`
  
Exporting Model:-  
`python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/ssd_resnet152_v1_fpn/pipeline.config --trained_checkpoint_dir models/ssd_resnet152_v1_fpn --output_directory exported-models/my_model`

