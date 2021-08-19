/**
 * @file main.cpp
 * @brief Sample vibration demo application using Edge Impulse neural network
 *
 * Copyright (c) 2021 Edge Impulse
 * Copyright (c) 2021 Laird Connectivity
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/******************************************************************************/
/* Includes                                                                   */
/******************************************************************************/
#include <zephyr.h>
#include <drivers/sensor.h>
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/dsp/numpy.hpp"

/******************************************************************************/
/* Local Constant, Macro and Type Definitions                                 */
/******************************************************************************/
/* Number of times to check */
#define CHECK_BUCKETS 20

/* Minimum number of pass-points needed to declare a winning frequency */
#define MIN_BUCKETS (CHECK_BUCKETS / 2)

/* Minimum number of pass-points needed to declare an error frequency */
#define ERR_BUCKETS (CHECK_BUCKETS / 4)

/* Minimum number of buckets that must have ERR_BUCKETS entries or more before declaring result as invalid */
#define ERR_BUCKETS_FAIL_COUNT 3

/* Minimum number of failures in a row before throwing an error back */
#define ERR_FAILS_IN_ROW 3

#define ACCEL_ARRAY_X 0
#define ACCEL_ARRAY_Y 1
#define ACCEL_ARRAY_Z 2
#define ACCEL_ARRAY_SIZE 3

#if defined(CONFIG_APP_AXIS_X_ENABLED) && defined(CONFIG_APP_AXIS_Y_ENABLED) && defined(CONFIG_APP_AXIS_Z_ENABLED)
#define AXIS_ENABLED 3
#elif (!defined(CONFIG_APP_AXIS_X_ENABLED) && defined(CONFIG_APP_AXIS_Y_ENABLED) && defined(CONFIG_APP_AXIS_Z_ENABLED)) || \
      (defined(CONFIG_APP_AXIS_X_ENABLED) && !defined(CONFIG_APP_AXIS_Y_ENABLED) && defined(CONFIG_APP_AXIS_Z_ENABLED)) || \
      (defined(CONFIG_APP_AXIS_X_ENABLED) && defined(CONFIG_APP_AXIS_Y_ENABLED) && !defined(CONFIG_APP_AXIS_Z_ENABLED))
#define AXIS_ENABLED 2
#elif defined(CONFIG_APP_AXIS_X_ENABLED) || defined(CONFIG_APP_AXIS_Y_ENABLED) || defined(CONFIG_APP_AXIS_Z_ENABLED)
#define AXIS_ENABLED 1
#else
#error "At least one axis must be enabled for the application to work"
#endif

#if AXIS_ENABLED != EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME
#error "The enabled axis must match axis which were enabled when the impulse was trained"
#endif

/******************************************************************************/
/* Local Data Definitions                                                     */
/******************************************************************************/
const static int64_t sampling_freq = EI_CLASSIFIER_FREQUENCY; /* in Hz */
static int64_t time_between_samples_us = (1000000 / (sampling_freq - 1));
static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

static uint8_t times[EI_CLASSIFIER_LABEL_COUNT];
static uint8_t runs = 0;
static uint8_t prev_fails = 0;
static uint32_t total_runs = 1;
static uint16_t run_time_dsp = 0;
static uint16_t run_time_classification = 0;

/* Array of entries that are considered good.
 * This corrisponds to 10Hz, 20Hz, 30Hz, 40Hz and 50Hz outputs
 */
const static uint8_t good_points[] = {
	1,
	3,
	4,
	5,
	6
};

/******************************************************************************/
/* Global Function Definitions                                                */
/******************************************************************************/
void main(void)
{
	struct k_timer next_val_timer;
	struct sensor_value accel[ACCEL_ARRAY_SIZE];

	/* Output immediately without buffering */
	setvbuf(stdout, NULL, _IONBF, 0);

	/* Find accelerometer driver instance */
	const struct device *iis2dlpc = device_get_binding(DT_LABEL(DT_INST(0, st_lis2dh)));
	if (iis2dlpc == NULL) {
		printf("Could not get IIS2DLPC device\n");
		return;
	}

	k_timer_init(&next_val_timer, NULL, NULL);

	while (1) {
		for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) {
			/* Start a timer that expires when we need to grab the next value */
			k_timer_start(&next_val_timer, K_USEC(time_between_samples_us), K_NO_WAIT);

			/* Perform a reading of the sensor data and retrieve it */
			if (sensor_sample_fetch(iis2dlpc) < 0) {
				printf("IIS2DLPC Sensor sample update error\n");
				return;
			}

			sensor_channel_get(iis2dlpc, SENSOR_CHAN_ACCEL_XYZ, accel);

			/* Move data from sensor result to buffer */
			size_t current_index = ix;

#if defined(CONFIG_APP_AXIS_X_ENABLED)
			features[current_index] = sensor_value_to_double(&accel[0]);
			++current_index;
#endif
#if defined(CONFIG_APP_AXIS_Y_ENABLED)
			features[current_index] = sensor_value_to_double(&accel[1]);
			++current_index;
#endif
#if defined(CONFIG_APP_AXIS_Z_ENABLED)
			features[current_index] = sensor_value_to_double(&accel[2]);
#endif

			if ((ix + EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME) < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
				/* Busy loop until next value should be grabbed */
				while (k_timer_status_get(&next_val_timer) <= 0);
			}
		}

		/* Create signal from features frame */
		ei_impulse_result_t result = { 0 };
		signal_t signal;
		numpy::signal_from_buffer(features, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

		/* Classify set of readings */
		EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

		if (res != 0) {
			printf("error: run_classifier returned %d\n", res);
			return;
		}

		/* Find largest index and value */
		float largest = 0;
		uint8_t largest_index = 0;
		for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
			if (result.classification[ix].value > largest) {
				largest = result.classification[ix].value;
				largest_index = ix;
			}
		}

		++times[largest_index];
		++runs;
		run_time_dsp += result.timing.dsp;
		run_time_classification += result.timing.classification;

#if defined(CONFIG_APP_OUTPUT_READABLE)
		printf("\rRun #%d loop %d of %d...", total_runs, runs, CHECK_BUCKETS);
#endif

		if (runs >= CHECK_BUCKETS) {
			/* Minimum number of buckets for output reached */
			bool high_fail = false;
			bool single_fail = false;
			bool good_pass = false;
			bool bad_pass = false;
			uint8_t entries_over_min = 0;
			uint8_t entries_over_err = 0;
			uint8_t max_id = 0;
			uint8_t max_val = 0;
			uint8_t max_dup = 1;

#if defined(CONFIG_APP_OUTPUT_DELIMITED)
			printf(">");
#else
			printf("\r\nDetections:\r\n");
#endif

			for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
					if (times[ix] >= MIN_BUCKETS) {
						++entries_over_min;
					}
					if (times[ix] >= ERR_BUCKETS) {
						++entries_over_err;
					}
					if (times[ix] >= max_val) {
						if (times[ix] == max_val) {
							++max_dup;
						} else {
							max_id = ix;
							max_val = times[ix];
							max_dup = 1;
						}
					}

#if defined(CONFIG_APP_OUTPUT_READABLE)
				printf("\t%s: %d\n",
				       result.classification[ix].label,
				       times[ix]);
#else
				printf("%d,", times[ix]);
#endif
			}

#if defined(CONFIG_APP_OUTPUT_READABLE)
			printf("Results:\r\n");
#endif

			if (entries_over_min == 0) {
				/* No entry with minimum number of points,
				 * cannot determine frequency, record as error
				 */
				++prev_fails;
#if defined(CONFIG_APP_OUTPUT_READABLE)
				printf("\tDetection failure (no winning entry)\r\n");
#endif
				if (prev_fails >= ERR_FAILS_IN_ROW) {
#if defined(CONFIG_APP_OUTPUT_READABLE)
					printf("\tHigh failure rate\r\n");
#endif
					high_fail = true;
				} else {
					single_fail = true;
				}
			} else if (entries_over_err >= ERR_BUCKETS_FAIL_COUNT) {
				/* Unable to determine frequency of
				 * oscillation, record as error
				 */
				++prev_fails;
#if defined(CONFIG_APP_OUTPUT_READABLE)
				printf("\tDetection failure (fail bucket count reached)\r\n");
#endif
				if (prev_fails >= ERR_FAILS_IN_ROW) {
#if defined(CONFIG_APP_OUTPUT_READABLE)
					printf("\tHigh failure rate\r\n");
#endif
					high_fail = true;
				} else {
					single_fail = true;
				}
			} else {
				/* Frequency determined */
				uint8_t i = 0;
				while (i < (sizeof(good_points)/sizeof(good_points[0]))) {
					if (good_points[i] == max_id) {
						/* Low frequency - class as good */
#if defined(CONFIG_APP_OUTPUT_READABLE)
						printf("\tLow frequency (%s) - good\r\n",
						       result.classification[max_id].label);
#endif
						good_pass = true;
						break;
					}
					++i;
				}

				if (i == (sizeof(good_points)/sizeof(good_points[0]))) {
					/* High frequency - class as bad */
#if defined(CONFIG_APP_OUTPUT_READABLE)
					printf("\tHigh/stationary frequency (%s) - bad\r\n",
					       result.classification[max_id].label);
#endif
					bad_pass = true;
				}
			}

#if defined(CONFIG_APP_OUTPUT_READABLE)
			printf("\t%d entr%s over winning minimum\r\n"
			       "\t%d entr%s over error minimum\r\n"
			       "\t%d entr%s with maximum value\r\n",
			       entries_over_min,
			       (entries_over_min == 1 ? "y" : "ies"),
			       entries_over_err,
			       (entries_over_err == 1 ? "y" : "ies"),
			       max_dup, (max_dup == 1 ? "y" : "ies"));
			printf("Run time:\r\n\tDSP: %dms\r\n"
			       "\tClassification: %dms\r\n",
			       run_time_dsp, run_time_classification);
			printf("\r\n");
#else
			printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d<",
			       entries_over_min, entries_over_err, max_id,
			       max_val, max_dup, good_pass, bad_pass,
			       high_fail, single_fail, run_time_dsp,
			       run_time_classification);
#endif

			memset(times, 0, sizeof(times));
			runs = 0;
			run_time_dsp = 0;
			run_time_classification = 0;
			++total_runs;
		}
	}
}
