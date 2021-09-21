from datetime import datetime,timedelta
import base64
import time
import numpy as np
import os
from annoy import AnnoyIndex
from keras.models import load_model
from google.cloud import storage 
import json
import time
import requests
import cv2

import pathlib
import time
from scipy.spatial.distance import cosine
import dlib

from fastapi import Request, FastAPI
import uvicorn
from starlette.responses import Response
from collections import Counter ,defaultdict
#tf.get_logger().setLevel('ERROR')

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]

storage_client=storage.Client()

main_dir = str(pathlib.Path(__file__).absolute().parent)

#load models
shape_predictor = os.path.join(main_dir,'model_files','shape_predictor_5_face_landmarks.dat')
dlib_face_recognition_model = os.path.join(main_dir,'model_files','dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor)
facerec = dlib.face_recognition_model_v1(dlib_face_recognition_model)


def download_load_annoy_json(temp_dir,model_bucket):
  try:
      local_ann_index_file_path = os.path.join(temp_dir, os.path.basename(cloud_ann_index_file_path))
      print("local_ann_index_file_path=",local_ann_index_file_path)
      print("cloud_ann_index_file_path=",cloud_ann_index_file_path)
      print("os.path.basename(cloud_ann_index_file_path) =",os.path.basename(cloud_ann_index_file_path))
      download_blob(model_bucket,cloud_ann_index_file_path,local_ann_index_file_path)
      print('Model downloaded')                        
      sequence_in_index_json_file_local_path = os.path.join(temp_dir,os.path.basename(sequence_in_index_json_file))
      print("sequence_in_index_json_file_local_path=",sequence_in_index_json_file_local_path)
      download_blob(model_bucket, sequence_in_index_json_file, sequence_in_index_json_file_local_path)
      print('json file downloaded')
      print('Loading annoy model')
      u = AnnoyIndex(128, 'angular')
      u.load(local_ann_index_file_path)
      print('Reading sequence json file')
      with open(sequence_in_index_json_file_local_path) as json_file:
        filenames = json.load(json_file)
      return u,filenames
    
  except Exception as e:
    print('Exception in download_load_annoy_json : ',e)
    return None,None



def image_search(u,image_array):
  start = time.time()
  try:
    if len(image_array) != 0:  
        # extracting features
        print('extracting embedding features')
        features = get_embedding(model, image_array)
        print('evaluating output')
        # evaluating output
        indices, distances = u.get_nns_by_vector(features, 5, include_distances=True)
        print("Indices :",indices)
        print("Distances :",distances)
        num_features = len(distances)
        print('filenames : ',filenames)
        if len(distances) > 0 :
            #name = filenames[indices[0]].split('/')[-3]
            name2 = filenames[indices[0]]
            distance = distances[0]
            end = time.time()
            time_taken = end - start
            print("Complete Name :",name2)
            print(name2,distance)
            return (name2,distance)
        else:
            print('Not found')
            return 0
  except Exception as e:
    print('Exception in image_search : ',e)
    #u,filenames = download_load_annoy_json(temp_dir,model_bucket)	
	
def load_settings():
  #sheet_url="https://script.google.com/a/bioenabletech.com/macros/s/AKfycbz6mnBNf__46yiIo6WEA0ljkxRBLz5nrQ5ccLQcAA/exec?"
  #data=requests.get(sheet_url+"page=matching").json()
  #print("settings_data=",data)
  image_save_bucket='c_function_test_bucket2'
  request_save_bucket='c_function_test_bucket3'
  default_folder='face_registration'
  default_organization='organization_1'
  default_project='project_1'
  c_function_url="https://asia-south1-cityairapp.cloudfunctions.net/face-match3"
  default_user_id="user_id"
  storage_path_name='storage_path'
  organization_name='organization'
  project_name='project'
  transaction_id_name='transaction_id'
  user_id_name='user_id'
  sleep_time=0
  return image_save_bucket,request_save_bucket,default_folder,default_organization,default_project,c_function_url,default_user_id,storage_path_name,organization_name,project_name,transaction_id_name,user_id_name,sleep_time

def upload_json(bucket, destination_jsonfile_name, result_json):
  bucket.blob(destination_jsonfile_name).upload_from_string(data=json.dumps(result_json),content_type='application/json')
  print('Data uploaded to {}.'.format(destination_jsonfile_name))

def download_blob(bucket, source_blob_name, destination_file_name):
   try:
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))
   except Exception as e:
        print("Exception in download_blob ",e)

def get_image_array(request_json):
  try:
    content=request_json['croppedImg']
    image_string_bytes=content.encode("utf8").split(b";base64,")[1] #converts image data to base64 bytes image
    im_bytes = base64.b64decode(image_string_bytes) #converts base 64 bytes to bytes image
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR) #converts to 3D Numpy array array
  except Exception as e:
    print("Exception in get_image_array :",e)
  return img

def get_embedding(model,face_pixels):
    image = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)
    img1_detection = detector(image, 1)
    img1_shape = sp(image, img1_detection[0])
    img1_aligned = dlib.get_face_chip(image, img1_shape)            
    img1_representation = facerec.compute_face_descriptor(img1_aligned)            
    img1_representation = np.array(img1_representation)
    return img1_representation

def check_files(request_json,timestamp):
  settings_start=time.time()
  image_save_bucket,request_save_bucket,default_folder,default_organization,default_project,c_function_url,default_user_id,storage_path_name,organization_name,project_name,transaction_id_name,user_id_name,sleep_time=load_settings()
  settings_end=time.time()
  print("time taken for settings function =",settings_end-settings_start)
  if storage_path_name in request_json:
    try:
      upload_bucket=request_json[storage_path_name].split('/')[0]
      main_folder=request_json[storage_path_name].split('/')[1]
    except:
      return {'transaction_id':timestamp,'status':'Failure','datetime':str(datetime.now()),"message":"invalid storage path.Should be in the form bucket_name/main_foldername"}
  else :
    upload_bucket=request_save_bucket
    main_folder=default_folder
  if organization_name in request_json:
    folder_name=request_json[organization_name]
  else :
    folder_name=default_organization
  if project_name in request_json:
    project_name=request_json[project_name]
  else :
    project_name=default_project
  if user_id_name in request_json:
    user_id=request_json[user_id_name]
  else : 
    user_id=default_user_id
  if transaction_id_name in request_json:
    transaction_id=timestamp
  else :
    transaction_id=""
  return image_save_bucket,request_save_bucket,main_folder,folder_name,project_name,user_id,transaction_id

def handle_request(request,payload):
  try:
    start=time.time()
    json_dict={}
    settings_dict={}
    list1=[]
    print('temp_dir: ',os.listdir(temp_dir))
    print("request method=",request.method)
    settings_dict['method']=request.method
    for head in request.headers:
        list1.append(head)
    json_dict['request_headers']=list1
    settings_dict['url']="https://asia-south1-cityairapp.cloudfunctions.net/face-match3"
    settings_dict['datetime']=str(datetime.now())
    timestamp=str(datetime.now()).replace("-","").replace(":","").replace(".","-").replace(" ","T")
    settings_dict['transaction_id']=timestamp
    json_dict['settings']=settings_dict
    print("request method=",request.method)
    request_method_time=time.time()
    print("time taken to come to request method=",request_method_time-start)
    if (request.method=='POST'):
      content_type = request.headers['Content-Type']
      print("content type :",content_type)
      if content_type == 'application/json':
        request_json = payload
        #print("request_json=",request_json)
        if (request_json!=None):
          if (len(request_json)!=0):
            json_dict['data']=request_json
            check_file_start=time.time()
            image_save_bucket, request_save_bucket, main_folder, folder_name, project_name, user_id, transaction_id = check_files(request_json,timestamp)
            check_file_end=time.time()
            print("check file time=",check_file_end-check_file_start)
            #storage_path=os.path.join(main_folder,folder_name,project_name,user_id,transaction_id)
            #request_save_bucket_obj = storage_client.get_bucket(request_save_bucket)
            #upload_json(request_save_bucket_obj,os.path.join(storage_path,"request.json"),json_dict)
            request_upload=time.time()
            print("Time taken for request_json_upload :",request_upload-check_file_end)
            #content=request_json['croppedImg']
            #print("taken croppedImg")
            #data=content.encode("utf8").split(b";base64,")[1]
            #image_path2='/tmp/image1-{}.png'.format(str(timestamp))
            #with open(image_path2, "wb") as fp:
            #        fp.write(base64.decodebytes(data))          
            #print("stored croppedImg to location {}".format(image_path2))
            #storage_path='test_images'
            #source_bucket = storage_client.get_bucket('c_function_test_bucket3')
            #source_bucket.blob(os.path.join(storage_path,image_path2.split('/')[-1])).upload_from_filename(image_path2)
            image_array=get_image_array(request_json)
            return image_array,timestamp
          else:
            return {'status':'Failure','transaction_id':timestamp,'datetime':str(datetime.now()),'message':'Invalid Json'},timestamp
        else:
          return {'status':'Failure','transaction_id':timestamp,'datetime':str(datetime.now()),'message':'json_data=None'},timestamp
    if (request.method=='OPTIONS'):
      return 'OPTIONS',''

  except Exception as e:
    print("Exception in handle request_function",e)

def download_registered_files():
  timestamp=os.path.getmtime(os.path.join(temp_dir,os.path.basename(cloud_ann_index_file_path)))
  dt_object = datetime.fromtimestamp(timestamp)
  current_time=datetime.now()
  print("dt_object =", dt_object)
  print("current time=",current_time)
  print("Time Difference=",current_time-dt_object)
  if current_time-dt_object>=timedelta(minutes=1):
    global u,filenames
    u,filenames = download_load_annoy_json(temp_dir,model_bucket)
    print("Called download_load_annoy_json function")



# download facenet model
print('global -- download facenet model')
#download_facenet_model(model_path)
# load facenet model
model = "load_model(model_path)"
temp_dir = os.path.join(main_dir,'temp_folder')
os.makedirs(temp_dir,exist_ok=True)
# download and load annoy model
MODEL_BUCKET = 'faces-out-dlib'
model_bucket = storage_client.get_bucket(MODEL_BUCKET)

cloud_ann_index_file_path = "index.ann"
sequence_in_index_json_file = "person_names.json"

u,filenames = download_load_annoy_json(temp_dir,model_bucket)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def face_matching(request: Request):
    print("within face_matching function")
    start=time.time()
    payload = await request.json()
    #print("payload=",payload)
    if not payload:
        msg = "no message received"
        print(f"error: {msg}")
        return f"Bad Request: {msg}", 400 
    try:
        image_array,timestamp=handle_request(request,payload)
        if image_array!='OPTIONS':
          try:
            image_array_time=time.time()
            print("Time taken for getting image_array=",image_array_time-start)
            name,distance = image_search(u,image_array)
            if distance<=0.30:
              output = name + ' (' + str(round(distance, 2))+')'
            else:
              output = "unknown_person" + ' (' + str(round(distance, 2))+')'
            prediction_time=time.time()
            print("Time taken for prediction =",prediction_time-image_array_time)
            print("Time taken for gcloud function =",prediction_time-start)
            download_registered_files()
            return {'status':'Success','transaction_id':timestamp,'datetime':str(datetime.now()),'message': output}
            
          except Exception as e:
            print("Exception occured after handle_request",e)
            return image_array
            download_registered_files()

          
    except Exception as e:
        print("Exception in handle request",e)

if __name__ == '__main__':
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8000
    uvicorn.run(app,host = '0.0.0.0',port=PORT)             
      


