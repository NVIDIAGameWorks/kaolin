const path = require('path');

module.exports = function (env, argv) {
    const mode = (argv && argv.mode) || 'production';

    // Configuration for building dash components
    const dashComponentsConfig = {
        name: 'dash-components',
        entry: {
            kaolin: path.join(__dirname, 'kaolin/visualize/dash/components/src/ts/index.ts')
        },
        output: {
            path: path.join(__dirname, 'kaolin/visualize/dash/components'),
            filename: 'autogen/[name].js',
            library: '[name]',
            libraryTarget: 'umd',
            globalObject: 'this'
        },
        mode,
        target: 'web',
        devtool: 'source-map',
        externals: [{
            react: {
                commonjs: 'react',
                commonjs2: 'react',
                amd: 'react',
                umd: 'react',
                root: 'React',
            },
            'react-dom': {
                commonjs: 'react-dom',
                commonjs2: 'react-dom',
                amd: 'react-dom',
                umd: 'react-dom',
                root: 'ReactDOM',
            },
            'playcanvas': {
                commonjs: 'playcanvas',
                commonjs2: 'playcanvas',
                amd: 'playcanvas',
                umd: 'playcanvas',
                root: 'pc',
            },
            // 'three': {
            //     commonjs: 'three',
            //     commonjs2: 'three',
            //     amd: 'three',
            //     umd: 'three',
            //     root: 'THREE',
            // }
        }
//        function ({ context, request }, callback) {
//            console.log('🍓-----> Externalizing request: ' + request);
//            if (/playcanvas$/.test(request) || /three$/.test(request)) {
//                console.log({context});
//            }
//            if (/^playcanvas$/.test(request)) {
//                console.log("We will actually externalize!");
//                return callback(null, 'playcanvas', 'module');
//            }
//            // Externalize to a commonjs module using the request path
//            //return callback(null, 'commonjs ' + request);
//
//            // Continue without externalizing the import
//            callback();
//        }
        ],
        resolve: {
            extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'],
            alias: {
                'three': path.resolve(__dirname, 'node_modules/three'),
            },
        },
        module: {
            rules: [
                {
                    test: /\.tsx?$/,
                    use: 'ts-loader',
                    exclude: /node_modules/,
                },
                {
                    test: /\.css$/,
                    use: [
                        {
                            loader: 'style-loader',
                            options: {insert: 'head'},
                        },
                        {
                            loader: 'css-loader',
                        },
                    ],
                },
            ]
        }
    };

    return dashComponentsConfig;
};
