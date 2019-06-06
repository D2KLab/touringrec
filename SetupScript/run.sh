for i in 0.001 0.01 0.1 0.3 0.4 0.5 0.6
do
python -W ignore Setup.py --epochs 100 --ncomponents 100 --lossfunction warp-kos --learningrate 0.1 --learningschedule adadelta --mfk 300 --useralpha 1e-6 --itemalpha 1e-6
done

for i in 100 200 300 400
do
python -W ignore Setup.py --epochs 200 --ncomponents $i --lossfunction warp --learningrate 0.1 --learningschedule adadelta --mfk 300 --useralpha 1e-6 --itemalpha 1e-6 --rho 1 --epsilon 0 --maxsampled 20
done

for i in "logistic" "bpr" "warp" "warp-kos"
do
python -W ignore Setup.py --epochs 200 --ncomponents 300 --lossfunction $i --learningrate 0.1 --learningschedule adadelta --mfk 300 --useralpha 1e-6 --itemalpha 1e-6 --rho 1 --epsilon 0 --maxsampled 20
done
