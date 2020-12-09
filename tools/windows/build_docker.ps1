$ErrorActionPreference = 'Stop'

$script_root = split-path -parent $MyInvocation.MyCommand.Definition
$build_root = $(split-path -parent $(split-path -parent $script_root))

$image_name = "kaolin-windows"
if ("$Args[0]" -ne "") {
    $image_name, $Args = $Args
}

cd $build_root

docker build --tag $image_name -f tools\windows\Dockerfile $Args .
if (-not($?)) {
    exit $LASTEXITCODE
}

docker run $image_name type conda_build.txt
if (-not($?)) {
    exit $LASTEXITCODE
}

if (Test-Path artifacts) {
    Remove-Item -Path artifacts -Recurse -Confirm:$false -Force
}
docker run --rm --volume "${build_root}:c:\kaolin\mnt" $image_name Copy-Item -Recurse dist\ mnt\artifacts

exit $LASTEXITCODE
