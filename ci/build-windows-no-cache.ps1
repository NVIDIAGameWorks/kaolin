$ErrorActionPreference = 'Stop'

$script_root = split-path -parent $MyInvocation.MyCommand.Definition

$image_name = "kaolin-windows"
if ("$env:CI_REGISTRY_IMAGE" -ne "") { 
    docker login -u $env:CI_REGISTRY_USER -p $env:CI_REGISTRY_PASSWORD $env:CI_REGISTRY
    if (-not($?)) {
        exit $LASTEXITCODE
    }
    $image_name = "$env:CI_REGISTRY_IMAGE/kaolin-windows:$env:CI_COMMIT_REF_SLUG"
}

& $script_root\..\tools\windows\build_docker.ps1 $image_name --no-cache
if (-not($?)) {
    exit $LASTEXITCODE
}

if ("$env:CI_REGISTRY_IMAGE" -ne "") { 
    docker push $image_name
}

exit $LASTEXITCODE

