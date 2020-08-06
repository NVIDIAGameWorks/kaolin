'use strict';
let broadlink = require('broadlinkjs-sm');
let fs = require('fs');

var b = new broadlink();

b.discover();

b.on("deviceReady", (dev) => {
	if (dev.type == "MP1") {
		var status = [];

		console.clear();
		console.log("Check Power...");
		dev.check_power();

		dev.on("mp_power", (status_array) => {
			status = status_array;

			console.log("");
			console.log("Switch Power -> " + status_array[0]);
			console.log("Shield Power -> " + status_array[1]);
			console.log("Pioneer Power -> " + status_array[2]);
			console.log("Amlogic Power -> " + status_array[3]);
			console.log(status_array);
		});

		setTimeout(function() {
			// Array index + 1
			dev.set_power(4,0);
			console.log("");
		}, 1000);
	} else {
		console.log(dev.type + "@" + dev.host.address + " found... not MP1!");
		dev.exit();
	}
});