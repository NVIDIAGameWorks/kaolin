// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Leveled console logger for Kaolin internal or user client-side code.
 *
 * Typical usage:
 * ```javascript
 * kaolin.logging.setLogLevel(kaolin.logging.LogLevel.INFO);
 *
 * // log at error / warn / info / debug
 * kaolin.logging.logger.error('problem!');
 * 
 * // Or custom logger
 * let log = new kaolin.logging.ConsoleLogger('my-app');
 * log.setLogLevel(kaolin.logging.LogLevel.INFO);
 * log.info('hello world');
 * ```
 *
 * @module
 */


/** Log levels for the console logger. */
export enum LogLevel {
    DEBUG = 10,
    INFO = 20,
    WARN = 30,
    ERROR = 40
}

/** Named logger that routes messages to the browser console based on a configurable minimum level. */
export class ConsoleLogger {
    private name: string;
    private logLevel: LogLevel;

    constructor(name: string = null, logLevel: LogLevel = LogLevel.DEBUG) {
        this.name = name;
        this.logLevel = logLevel;
    }

    setLogLevel(level: LogLevel): void {
        this.logLevel = level;
    }

    getLogLevel(): LogLevel {
        return this.logLevel;
    }

    private formatMessage(message: string): string {
        let d = new Date();
        let pfix = this.name ? this.name + " - " : "";
        let msg = pfix + d.getMinutes() + ":" + d.getSeconds() + ":" + d.getMilliseconds() + "  " + message;
        return msg;
    }

    error(message: string, ...args: any[]): void {
        if (this.logLevel <= LogLevel.ERROR) {
            console.error(this.formatMessage(message), ...args);
        }
    }

    warn(message: string, ...args: any[]): void {
        if (this.logLevel <= LogLevel.WARN) {
            console.warn(this.formatMessage(message), ...args);
        }
    }

    info(message: string, ...args: any[]): void {
        if (this.logLevel <= LogLevel.INFO) {
            console.info(this.formatMessage(message), ...args);
        }
    }

    debug(message: string, ...args: any[]): void {
        if (this.logLevel <= LogLevel.DEBUG) {
            console.log(this.formatMessage(message), ...args);
        }
    }
}

/** Module-level logger singleton used across Kaolin client-side code. */
export const logger = new ConsoleLogger("kaolin", LogLevel.DEBUG);

/**
 * Set the minimum log level for the module-level {@link logger} singleton.
 *
 * @param level - Minimum level; messages below this level are suppressed.
 */
export function setLogLevel(level: LogLevel): void {
    logger.setLogLevel(level);
}

/**
 * Return the current log level of the module-level {@link logger} singleton.
 *
 * @returns The active {@link LogLevel}.
 */
export function getLogLevel(): LogLevel {
    return logger.getLogLevel();
}