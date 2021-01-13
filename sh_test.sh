echo 'please input the gene number...'
echo 'python3 /home/yuxuan/dp_m6a_org/main.py -gene eif3a -condition Full -length 125 -mode CNN'
read x

echo $x

echo 'please input the condition...'
read y
echo $y


echo 'please read the length...'
read z
echo $z


echo 'please input the mode...'
read j
echo $j

python3 /home/yuxuan/dp_m6a_org/main.py -gene $x -condition $y -length $z -mode $j