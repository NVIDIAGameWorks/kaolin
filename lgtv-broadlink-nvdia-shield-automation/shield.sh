adb connect 192.168.1.106 > /dev/null

if [ ! -d ~/.homebridgestates ]
then
    mkdir ~/.homebridgestates
fi

if [ $1 = "keyevent" ]
then
	adb shell input keyevent $2
elif [ $1 = "wake" ]
then
    adb shell input keyevent KEYCODE_WAKEUP
elif [ $1 = "sleep" ]
then
    adb shell input keyevent KEYCODE_SLEEP
elif [ $1 = "sleepstatus" ]
then
    status=$(adb shell "dumpsys power | grep mHoldingDisplay  | cut -d = -f 2");
    echo $status
fi