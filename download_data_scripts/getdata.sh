for (( i=3; i<=3; i++ ))
do 
  wget -x -c --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://www.kaggle.com/nih-chest-xrays/data/downloads/images_00$i.zip
done
