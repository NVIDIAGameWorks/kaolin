if [ "$#" -ne 1 ]; then
    printf "Illegal number of parameters provided\n" >&2
    exit 2
fi

case $1 in

  "1.8.0")
    echo "0.9.0"
    ;;

  "1.8.1")
    echo "0.9.1"
    ;;

  "1.9.0")
    echo "0.10.0"
    ;;

  "1.9.1")
    echo "0.10.1"
    ;;

  "1.10.0")
    echo "0.11.0"
    ;;

  "1.10.1")
    echo "0.11.2"
    ;;

  "1.10.2")
    echo "0.11.3"
    ;;

  "1.11.0")
    echo "0.12.0"
    ;;

  "1.12.0")
    echo "0.13.0"
    ;;

  "1.12.1")
    echo "0.13.1"
    ;;

  "1.13.0")
    echo "0.14.0"
    ;;

  *)
    printf "ERROR $1 unsupported torch version\n" >&2
    exit 3
    ;;
esac
