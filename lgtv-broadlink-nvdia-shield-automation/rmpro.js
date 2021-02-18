/*
	usage:

	When you want to learn a code, it might took several times before the correct code saved
		node rmpro.js learn input-name

	When you want to send the code
		node rmpro.js send input-name

	When you want to read the code
		node rmpro.js read input-name
*/

'use strict';
let broadlink = require('broadlinkjs');
let fs = require('fs');
let path = require('path');

var b = new broadlink();

if (process.argv.length < 3) {
	var _files = '',
		i = 0;

	console.log("Usage:\n", "\tnode rmpro send (command)", "\n");
	console.log("Available command:")
	fs.readdir("code/", (err, files) => {
		files.forEach(file => {
			if (i > 0) _files += ", ";
			_files += file.replace(".bin", "");
			i++;
		});

		console.log(_files);
	})
} else {
	var learn = process.argv[2];
	var file = process.argv[3];
	var _file = process.argv[4];

	if (learn == "read" || learn == "r") {
		// Buffer mydata
		function bufferFile(relPath) {
			return fs.readFileSync(path.join(__dirname, relPath)); // zzzz....
		}

		var data = bufferFile("code/" + file + ".bin");
		data = new Buffer.from(data, 'ascii').toString('hex');

		console.log("Code -> " + data);

		process.exit();
	} else if (learn == "convert" || learn == "c") {
		var data = Buffer.from(file, 'base64');

		console.log(data);

		fs.writeFile("code/" + _file + ".bin", data, function(err) {
			if(err) {
				return console.log(err);
			}

			console.log("The file was saved!");

			process.exit();
		});
	} else {
		b.on("deviceReady", (dev) => {
			if(dev.type == "RMPro") {
				// Buffer mydata
				function bufferFile(relPath) {
					return fs.readFileSync(path.join(__dirname, relPath)); // zzzz....
				}

				console.log("Connected -> " + dev.host.address)
				if (learn == "learn" || learn == "l") {
					console.log("Waiting for input ->", file);
					var timer = setInterval(function(){
						dev.checkData();
					}, 500);

					dev.on("rawData", (data) => {
						fs.writeFile("code/" + file + ".bin", data, function(err) {
							if(err) {
								return console.log(err);
							}

							console.log("The file was saved!");

							data = new Buffer.from(data, 'ascii').toString('hex');

							console.log("Code -> " + data);

							process.exit();
						});
					});

					dev.enterLearning();
				} else if (learn == "sendcode" || learn == "sc") {
					console.log("Sending data ->", file);

					data = new Buffer.from(file, 'ascii');
					dev.sendData(data);

					process.exit();
				} else {
					console.log("Sending data ->", file);
					dev.sendData(bufferFile("code/" + file + ".bin"));

					var timer = setInterval(function(){
						clearInterval(timer);
						process.exit();
					}, 500);
				}
			}
		});
	}

	b.discover();
}