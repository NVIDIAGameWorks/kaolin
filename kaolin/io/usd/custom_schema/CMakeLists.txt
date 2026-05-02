set(PXR_PREFIX )
set(PXR_PACKAGE custom_schema)

pxr_library(custom_schema
    INCLUDE_SCHEMA_FILES

    LIBRARIES
        # List dependency libraries

    CPPFILES
        # List non-schema cpp files

    PYMODULE_FILES
        __init__.py
)
