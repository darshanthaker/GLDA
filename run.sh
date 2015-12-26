echo "GLDA:"
echo "Num of worker threads: "
read NUM_WORKERS
echo "Use pre-existing model?: "
select yn in "y" "n"; do
    case $yn in
        y ) python GLDA.py $NUM_WORKERS 1 ; break;;
        n ) python GLDA.py $NUM_WORKERS 0 ; break;;
    esac
done
