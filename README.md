# ECE5424Project

#This is the initial Readme which will explain how everything works and is installed

#How to Install Llama
1
Install the Llama CLI
In your preferred environment run the command below:
Command
pip install llama-stack
Use -U option to update llama-stack if a previous version is already installed:
Command
pip install llama-stack -U
2
Find models list
See latest available models by running the following command and determine the model ID you wish to download:
Command
llama model list
If you want older versions of models, run the command below to show all the available Llama models:
Command
llama model list --show-all
3
Select a model
Select a desired model by running:
Command
llama model download --source meta --model-id  MODEL_ID
4
Specify custom URL
Llama 3.1: 405B & 8B
When the script asks for your unique custom URL, please paste the URL below
URL
https://llama3-1.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiMmRuMWdiZThkZHdhM3ZwczM1bTc0cWEwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTEubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3MzU5OTIxN319fV19&Signature=qvkxSVEncXipJbbJpx84QUMkFwY8jCnDuekZQ8tF-aX2nFkecAY2oQixQfWWg4KK6hJUoaPOisTzkmYy47PrBZPZvff6h7h4HCw00oM8R4AOAMkOHNH1p%7EtmgUr0ET%7Erdf-5NivC9I%7EQNkohaZDWkdfTYtclfz6wLmzpkk3luwK7DAwp9cAPz0DH7dNXFgt-ETcWhaLvsDtaWNIrEuKw4Ox%7EtYuCVJxGVJplsAkc9YQ9RdNHi-P9aRnwGZAt5lQwBN0vM9bTX-XkYBExEubNFTN5cU9ErjhjF34IQMO47QXXQeoN-B%7E-%7EXwji17zcJgM-TXwyHX14hMkdgv5TVpeZQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=954160630483037
