SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

if [ ! -d ~/.homebridgestates ]
then
	mkdir ~/.homebridgestates
fi

if [ $1 = "on" ]
then
	python "$SCRIPT_DIR/sendCode.py" $3
	echo $2 "is on"
	echo "This is flag switch on for" $2 > ~/.homebridgestates/${2}.flag
elif [ $1 = "off" ]
then
	python "$SCRIPT_DIR/sendCode.py" $3
	echo $2 "is off"
	rm ~/.homebridgestates/${2}.flag
else
	if [ -f ~/.homebridgestates/${2}.flag ]; then
	   echo "true"
	else
	   echo "false"
	fi
fi
