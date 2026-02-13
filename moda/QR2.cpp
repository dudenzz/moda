///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// QR2Solver - Quick R2 algorithm
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>  
#include <algorithm>
#include "Hypervolume.h"
#include "QR2.h"


namespace moda {

	R2Result* QR2Solver::Solve(DataSet* problem, QR2Parameters settings)
	{
		//initialize the problem
		prepareData(problem, settings);

		//call the starting callback
		StartCallback(*currentSettings, "QR2 Solver");

		//specific to R2 data prep
		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
			ReferencePoint.ObjectiveValues[j] = betterPoint->ObjectiveValues[j] * 1.1;
			worsePoint->ObjectiveValues[j] = -inf;
		}

		//calculate R2
		it0 = clock();
		R2Result* r = solveQR2(currentlySolvedProblem->points, *betterPoint, *worsePoint, settings.CalculateHV);
		r->type = Result::ResultType::R2;
		r->ElapsedTime = clock() - it0;
		r->FinalResult = true;
		//call the closing callback
		EndCallback(*currentSettings, r);

		//return the result
		return r;
	}


	inline DType powInt(DType x, short n) {
		DType s = x;
		short i;
		for (i = 1; i < n; i++)
			s *= x;
		return s;
	}








	DType QR2Solver::calculateR2(short h) {


		currentSettings->NumberOfObjectives;

		DType minMax = yMax[0];
		short k = 0;
		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {

			if (minMax > yMax[j]) {
				minMax = yMax[j];
				k = j;
			}
		}

		bool kUnique = true;
		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
			if (j != k && minMax > yMin[j]) {
				kUnique = false;
				break;
			}
		}



		if (kUnique) {

			if (k != h) {


				// Eq. 43 (44)
				DType e = 0;
				if (yMax[h] != inf) {
					e = 1;
					for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
						if (j != k && j != h) {
							e *= t[j];
						}
					}

					//DType e = tAllProduct / t[k] / t[h];

					DType yMaxProduct = yMax[k];
					DType yMinProduct = yMin[k];
					for (short i = 1; i < currentSettings->NumberOfObjectives; i++) {
						yMaxProduct *= yMax[k];
						yMinProduct *= yMin[k];
					}
					e *= (yMaxProduct - yMinProduct) / (currentSettings->NumberOfObjectives * yMax[h]);

					//e *= (powInt(yMax[k], NumberOfObjectives) - powInt(yMin[k], NumberOfObjectives)) / (NumberOfObjectives * yMax[h]);

				}

	

				return e;
			}
			else {
		

				// Eq. 46 (44)
				DType e = yMax[k];
				for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
					if (j != k) {
						e *= yMax[k] * t[j];
					}
				}

			

				return e;
			}
		}
		else {

			DType e = 0;
			short potentialMinsSize = 0;
			for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
				if (minMax >= yMin[j]) {
					potentialMins[potentialMinsSize++] = j;
				}
			}



			// sort
			for (short i = 0; i < potentialMinsSize - 1; i++) {
				for (short ii = i + 1; ii < potentialMinsSize; ii++) {
					if (yMin[potentialMins[i]] > yMin[potentialMins[ii]]) {
						short t = potentialMins[i];
						potentialMins[i] = potentialMins[ii];
						potentialMins[ii] = t;
					}
				}
			}

			// Fig. 4
			Range range;
			range.setJSize = 0;
			short rangesSize = 0;

			for (short i = 0; i < potentialMinsSize - 1; i++) {
				range.setJ[range.setJSize++] = potentialMins[i];
				if (yMin[potentialMins[i]] < yMin[potentialMins[i + 1]]) {
					range.yMin = yMin[potentialMins[i]];
					range.yMax = yMin[potentialMins[i + 1]];
					ranges[rangesSize++] = range;
				}
			}

			range.setJ[range.setJSize++] = potentialMins[potentialMinsSize - 1];
			range.yMin = yMin[potentialMins[potentialMinsSize - 1]];
			range.yMax = minMax;
			ranges[rangesSize++] = range;

			for (short ir = 0; ir < rangesSize; ir++) {
				range = ranges[ir];

				//cout << ">>> h=" << h << "\tir=" << ir << "\tyMin=" << range.yMin << "\tyMax=" << range.yMax << "\tsetJ=[";
				//for (short i = 0; i < range.setJSize; i++) cout << " " << range.setJ[i]; cout << " ]\tyMax=[";
				//for (short i = 0; i < NumberOfObjectives; i++) cout << " " << yMax[i]; cout << " ]\n";



				if (range.yMin == range.yMax) {


					// range IV in Fig. 4
					// Eq. 46 (44) with range limitation
					DType e1 = 1;
					for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
						if (j != h) {
							if (yMin[j] >= range.yMin) {
								e1 *= t[j];
							}
							else {
								e1 *= 1.0 / range.yMin - 1.0 / yMax[j];
							}
						}
					}

					e1 *= powInt(yMax[h], currentSettings->NumberOfObjectives);

					e += e1;



					continue;
				}

				if (range.setJSize == 1) {


					// range I in Fig. 4
					// Eq. 43 (44) with range limitation 
					k = range.setJ[0];

					DType e1 = 0;
					if (yMax[h] != inf) {
						e1 = 1;
						for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
							if (j != k && j != h) {
								e1 *= t[j];
							}
						}

						//e1 *= (powInt(range.yMax, NumberOfObjectives) - powInt(range.yMin, NumberOfObjectives)) / (NumberOfObjectives * yMax[h]);

						//DType e1 = tAllProduct / t[k] / t[h];

						DType yMaxProduct = range.yMax;
						DType yMinProduct = range.yMin;
						for (short i = 1; i < currentSettings->NumberOfObjectives; i++) {
							yMaxProduct *= range.yMax;
							yMinProduct *= range.yMin;
						}
						e1 *= (yMaxProduct - yMinProduct) / (currentSettings->NumberOfObjectives * yMax[h]);

						e += e1;
					}



					continue;
				}

				// range II or III in Fig. 4
				// Eq. 50 (44)
				short k1, k2, k3, k4, k5, k6;
				short j, i, ii, i1, i2, i3, i4, i5, i6;
				DType f1, f2;
				DType eMin, eMax;
				DType eBase = 1.0 / yMax[h];
				short setNotJSize = 0;
				for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
					if (j != h && std::find(range.setJ, range.setJ + range.setJSize, j) == range.setJ + range.setJSize) {
						setNotJSize++;
						eBase *= t[j];
					}
				}



				yMinPow[2] = range.yMin * range.yMin;
				yMinPow[3] = yMinPow[2] * range.yMin;
				yMinPow[4] = yMinPow[3] * range.yMin;
				yMinPow[5] = yMinPow[4] * range.yMin;
				yMinPow[6] = yMinPow[5] * range.yMin;
				yMinPow[7] = yMinPow[6] * range.yMin;
				yMinPow[8] = yMinPow[7] * range.yMin;
				yMinPow[9] = yMinPow[8] * range.yMin;
				yMinPow[10] = yMinPow[9] * range.yMin;

				yMaxPow[2] = range.yMax * range.yMax;
				yMaxPow[3] = yMaxPow[2] * range.yMax;
				yMaxPow[4] = yMaxPow[3] * range.yMax;
				yMaxPow[5] = yMaxPow[4] * range.yMax;
				yMaxPow[6] = yMaxPow[5] * range.yMax;
				yMaxPow[7] = yMaxPow[6] * range.yMax;
				yMaxPow[8] = yMaxPow[7] * range.yMax;
				yMaxPow[9] = yMaxPow[8] * range.yMax;
				yMaxPow[10] = yMaxPow[9] * range.yMax;


				DType e1 = 0;
				ii = setNotJSize + 2;
				DType f0 = 1.0 / ii;
				DType eMax0 = yMaxPow[ii] * f0;
				DType eMin0 = yMinPow[ii] * f0;

				for (i = 0; i < range.setJSize; i++) {
					k = range.setJ[i];
					eMax = eMax0;
					eMin = eMin0;
					ii = setNotJSize + 3;
					f1 = 1.0 / ii;

					for (i1 = 0; i1 < range.setJSize; i1++) {
						k1 = range.setJ[i1];
						if (yMax[k1] == inf) {
							continue;
						}

						if (k1 != k) {
							f2 = f1 / yMax[k1];
							eMax -= yMaxPow[ii] * f2;
							eMin -= yMinPow[ii] * f2;
						}
					}

					if (range.setJSize > 2) {
						ii = setNotJSize + 4;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 1; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								if (k1 != k && k2 != k) {
									f2 = f1 / (yMax[k1] * yMax[k2]);
									eMax += yMaxPow[ii] * f2;
									eMin += yMinPow[ii] * f2;
								}
							}
						}
					}

					if (range.setJSize > 3) {
						ii = setNotJSize + 5;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 2; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 1; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									if (k1 != k && k2 != k && k3 != k) {
										f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3]);
										eMax -= yMaxPow[ii] * f2;
										eMin -= yMinPow[ii] * f2;
									}
								}
							}
						}
					}

					if (range.setJSize > 4) {
						ii = setNotJSize + 6;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 3; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 2; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 1; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										if (k1 != k && k2 != k && k3 != k && k4 != k) {
											f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4]);
											eMax += yMaxPow[ii] * f2;
											eMin += yMinPow[ii] * f2;
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 5) {
						ii = setNotJSize + 7;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 4; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 3; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 2; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 1; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k) {
												f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5]);
												eMax -= yMaxPow[ii] * f2;
												eMin -= yMinPow[ii] * f2;
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 6) {
						ii = setNotJSize + 8;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 5; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 4; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 3; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 2; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize - 1; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											for (i6 = i5 + 1; i6 < range.setJSize; i6++) {
												k6 = range.setJ[i6];
												if (yMax[k6] == inf) {
													continue;
												}

												if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k && k6 != k) {
													f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5] * yMax[k6]);
													eMax += yMaxPow[ii] * f2;
													eMin += yMinPow[ii] * f2;
												}
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 7) {
						//cout << " *** Above limit ***\n";
						short count = 0;
						for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
							if (j != k && yMax[j] != inf) {
								count++;
							}
						}
						if (count > 6) {
							std::cout << " *** Not Implemented ***\n";
							exit(-100);
						}
					}

					e1 += (eMax - eMin) * eBase;
				}

				e += e1;


			}



			return e;
		}
	}


	DType QR2Solver::calculateR2(short h, Range& range) {


		short k;
		DType e = 0;
		if (range.yMin == range.yMax) {


			// range IV in Fig. 4
			// Eq. 46 (44) with range limitation
			if (yMax[h] != inf) {
				e = 1;
				for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
					if (j != h) {
						if (yMin[j] >= range.yMin) {
							e *= t[j];
						}
						else {
							e *= 1 / range.yMin - 1 / yMax[j];
						}
					}
				}
				if (e != 0) {
					e *= powInt(yMax[h], currentSettings->NumberOfObjectives);

				}
			}
		}
		else if (range.setJSize == 1) {


			// range I in Fig. 4
			// Eq. 43 (44) with range limitation 
			if (yMax[h] != inf) {
				e = 1;
				k = range.setJ[0];

				for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
					if (j != k && j != h) {
						e *= t[j];
					}
				}

				DType yMaxProduct = range.yMax;
				DType yMinProduct = range.yMin;
				for (short i = 1; i < currentSettings->NumberOfObjectives; i++) {
					yMaxProduct *= range.yMax;
					yMinProduct *= range.yMin;
				}
				e *= (yMaxProduct - yMinProduct) / (currentSettings->NumberOfObjectives * yMax[h]);

			}
		}
		else {
			// range II or III in Fig. 4
			// Eq. 50 (44)


			if (yMax[h] != inf) {
				short k1, k2, k3, k4, k5, k6, k7, k8;
				short j, i, ii, i1, i2, i3, i4, i5, i6, i7, i8;
				DType f1, f2;
				DType eMin, eMax;
				DType eBase = 1.0 / yMax[h];
				short setNotJSize = 0;

				for (j = 0; j < currentSettings->NumberOfObjectives; j++) {
					if (j != h && std::find(range.setJ, range.setJ + range.setJSize, j) == range.setJ + range.setJSize) {
						setNotJSize++;
						eBase *= t[j];
					}
				}



				yMinPow[2] = range.yMin * range.yMin;
				yMinPow[3] = yMinPow[2] * range.yMin;
				yMinPow[4] = yMinPow[3] * range.yMin;
				yMinPow[5] = yMinPow[4] * range.yMin;
				yMinPow[6] = yMinPow[5] * range.yMin;
				yMinPow[7] = yMinPow[6] * range.yMin;
				yMinPow[8] = yMinPow[7] * range.yMin;
				yMinPow[9] = yMinPow[8] * range.yMin;
				yMinPow[10] = yMinPow[9] * range.yMin;

				yMaxPow[2] = range.yMax * range.yMax;
				yMaxPow[3] = yMaxPow[2] * range.yMax;
				yMaxPow[4] = yMaxPow[3] * range.yMax;
				yMaxPow[5] = yMaxPow[4] * range.yMax;
				yMaxPow[6] = yMaxPow[5] * range.yMax;
				yMaxPow[7] = yMaxPow[6] * range.yMax;
				yMaxPow[8] = yMaxPow[7] * range.yMax;
				yMaxPow[9] = yMaxPow[8] * range.yMax;
				yMaxPow[10] = yMaxPow[9] * range.yMax;


				ii = setNotJSize + 2;
				DType f0 = 1.0 / ii;
				DType eMax0 = yMaxPow[ii] * f0;
				DType eMin0 = yMinPow[ii] * f0;

				for (i = 0; i < range.setJSize; i++) {
					k = range.setJ[i];
					eMax = eMax0;
					eMin = eMin0;
					ii = setNotJSize + 3;
					f1 = 1.0 / ii;

					for (i1 = 0; i1 < range.setJSize; i1++) {
						k1 = range.setJ[i1];
						if (yMax[k1] == inf) {
							continue;
						}

						if (k1 != k) {
							f2 = f1 / yMax[k1];
							eMax -= yMaxPow[ii] * f2;
							eMin -= yMinPow[ii] * f2;
						}
					}

					if (range.setJSize > 2) {
						ii = setNotJSize + 4;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 1; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								if (k1 != k && k2 != k) {
									f2 = f1 / (yMax[k1] * yMax[k2]);
									eMax += yMaxPow[ii] * f2;
									eMin += yMinPow[ii] * f2;
								}
							}
						}
					}

					if (range.setJSize > 3) {
						ii = setNotJSize + 5;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 2; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 1; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									if (k1 != k && k2 != k && k3 != k) {
										f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3]);
										eMax -= yMaxPow[ii] * f2;
										eMin -= yMinPow[ii] * f2;
									}
								}
							}
						}
					}

					if (range.setJSize > 4) {
						ii = setNotJSize + 6;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 3; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 2; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 1; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										if (k1 != k && k2 != k && k3 != k && k4 != k) {
											f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4]);
											eMax += yMaxPow[ii] * f2;
											eMin += yMinPow[ii] * f2;
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 5) {
						ii = setNotJSize + 7;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 4; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 3; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 2; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 1; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k) {
												f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5]);
												eMax -= yMaxPow[ii] * f2;
												eMin -= yMinPow[ii] * f2;
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 6) {
						ii = setNotJSize + 8;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 5; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 4; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 3; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 2; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize - 1; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											for (i6 = i5 + 1; i6 < range.setJSize; i6++) {
												k6 = range.setJ[i6];
												if (yMax[k6] == inf) {
													continue;
												}

												if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k && k6 != k) {
													f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5] * yMax[k6]);
													eMax += yMaxPow[ii] * f2;
													eMin += yMinPow[ii] * f2;
												}
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 7) {
						ii = setNotJSize + 9;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 6; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 5; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 4; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 3; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize - 2; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											for (i6 = i5 + 1; i6 < range.setJSize - 1; i6++) {
												k6 = range.setJ[i6];
												if (yMax[k6] == inf) {
													continue;
												}

												for (i7 = i6 + 1; i7 < range.setJSize; i7++) {
													k7 = range.setJ[i7];
													if (yMax[k7] == inf) {
														continue;
													}

													if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k && k6 != k && k7 != k) {
														f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5] * yMax[k6] * yMax[k7]);
														eMax -= yMaxPow[ii] * f2;
														eMin -= yMinPow[ii] * f2;
													}
												}
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 8) {
						ii = setNotJSize + 10;
						f1 = 1.0 / ii;

						for (i1 = 0; i1 < range.setJSize - 7; i1++) {
							k1 = range.setJ[i1];
							if (yMax[k1] == inf) {
								continue;
							}

							for (i2 = i1 + 1; i2 < range.setJSize - 6; i2++) {
								k2 = range.setJ[i2];
								if (yMax[k2] == inf) {
									continue;
								}

								for (i3 = i2 + 1; i3 < range.setJSize - 5; i3++) {
									k3 = range.setJ[i3];
									if (yMax[k3] == inf) {
										continue;
									}

									for (i4 = i3 + 1; i4 < range.setJSize - 4; i4++) {
										k4 = range.setJ[i4];
										if (yMax[k4] == inf) {
											continue;
										}

										for (i5 = i4 + 1; i5 < range.setJSize - 3; i5++) {
											k5 = range.setJ[i5];
											if (yMax[k5] == inf) {
												continue;
											}

											for (i6 = i5 + 1; i6 < range.setJSize - 2; i6++) {
												k6 = range.setJ[i6];
												if (yMax[k6] == inf) {
													continue;
												}

												for (i7 = i6 + 1; i7 < range.setJSize - 1; i7++) {
													k7 = range.setJ[i7];
													if (yMax[k7] == inf) {
														continue;
													}

													for (i8 = i7 + 1; i8 < range.setJSize; i8++) {
														k8 = range.setJ[i8];
														if (yMax[k8] == inf) {
															continue;
														}

														if (k1 != k && k2 != k && k3 != k && k4 != k && k5 != k && k6 != k && k7 != k && k8 != k) {
															f2 = f1 / (yMax[k1] * yMax[k2] * yMax[k3] * yMax[k4] * yMax[k5] * yMax[k6] * yMax[k7] * yMax[k8]);
															eMax += yMaxPow[ii] * f2;
															eMin += yMinPow[ii] * f2;
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}

					if (range.setJSize > 9) {
						//cout << " *** Above limit ***\n";
						short count = 0;
						for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
							if (j != k && yMax[j] != inf) {
								count++;
							}
						}
						if (count > 8) {
							std::cout << " *** Not Implemented ***\n";
							exit(-100);
						}
					}

					e += (eMax - eMin) * eBase;
				}
			}
		}



		return e;
	}


	DType QR2Solver::calculateR2Tentative(const Point& nadirPoint, const Point& point, const Point& idealPoint, bool calculateHV) {
		DType result = 0;



		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
			yMax[j] = ReferencePoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j];
			yMin[j] = ReferencePoint.ObjectiveValues[j] - std::min(point.ObjectiveValues[j], idealPoint.ObjectiveValues[j]);

			t[j] = 1 / yMin[j] - 1 / yMax[j];
		}

		DType r2;
		for (short h = 0; h < currentSettings->NumberOfObjectives; h++) {
			yMax[h] = yMin[h];
			r2 = calculateR2(h);
			yMax[h] = inf;

			result += r2;
		}



		return result;
	}


	DType QR2Solver::calculateR2Contribution(const Point& nadirPoint, const Point& point, const Point& idealPoint, bool calculateHV) {
		DType result = 0;



		DType minmax = std::numeric_limits<float>::max();
		short k = 0;
		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
			yMax[j] = ReferencePoint.ObjectiveValues[j] - nadirPoint.ObjectiveValues[j];
			yMin[j] = ReferencePoint.ObjectiveValues[j] - std::min(point.ObjectiveValues[j], idealPoint.ObjectiveValues[j]);

			t[j] = 1 / yMin[j] - 1 / yMax[j];

			if (minmax > yMax[j]) {
				minmax = yMax[j];
				k = j;
			}
		}

		bool kUnique = true;
		for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
			if (j != k && minmax > yMin[j]) {
				kUnique = false;
				break;
			}
		}

		if (kUnique) {

			result = -1.0 / currentSettings->NumberOfObjectives;

			for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
				if (j != k) {
					result *= t[j];
				}
			}

			result *= (powInt(yMax[k], currentSettings->NumberOfObjectives) - powInt(yMin[k], currentSettings->NumberOfObjectives));

		}
		else {

			short potentialMinsSize = 0;
			for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
				if (minmax >= yMin[j]) {
					potentialMins[potentialMinsSize++] = j;
				}
			}



			// sort
			for (short i = 0; i < potentialMinsSize - 1; i++) {
				for (short ii = i + 1; ii < potentialMinsSize; ii++) {
					if (yMin[potentialMins[i]] > yMin[potentialMins[ii]]) {
						short t = potentialMins[i];
						potentialMins[i] = potentialMins[ii];
						potentialMins[ii] = t;
					}
				}
			}

			// Fig. 4
			Range range;
			range.setJSize = 0;
			short rangesSize = 0;

			for (short i = 0; i < potentialMinsSize - 1; i++) {
				range.setJ[range.setJSize++] = potentialMins[i];
				if (yMin[potentialMins[i]] < yMin[potentialMins[i + 1]]) {
					range.yMin = yMin[potentialMins[i]];
					range.yMax = yMin[potentialMins[i + 1]];
					ranges[rangesSize++] = range;
				}
			}
			range.setJ[range.setJSize++] = potentialMins[potentialMinsSize - 1];
			range.yMin = yMin[potentialMins[potentialMinsSize - 1]];
			range.yMax = minmax;
			ranges[rangesSize++] = range;

			DType r2;
			for (short ir = 0; ir < rangesSize; ir++) {
				range = ranges[ir];



				DType yTmp;
				for (short h = 0; h < currentSettings->NumberOfObjectives; h++) {
					if (range.yMin < yMin[h]) {
						yTmp = yMax[h];
						yMax[h] = yMin[h];
						r2 = calculateR2(h, range);
						yMax[h] = yTmp;
						result += r2;
					}

					yTmp = yMin[h];
					short* it = std::find(range.setJ, range.setJ + range.setJSize, h);
					bool removed = false;
					if (it != range.setJ + range.setJSize) {
						*it = range.setJ[--range.setJSize];
						removed = true;
					}

					if (range.setJSize > 0) {
						yMin[h] = yMax[h];
						r2 = calculateR2(h, range);
						result -= r2;
						yMin[h] = yTmp;
					}

					if (removed) {
						range.setJ[range.setJSize++] = h;
					}
				}
			}

			DType yTmp;
			for (short h = 0; h < currentSettings->NumberOfObjectives; h++) {
				if (std::find(potentialMins, potentialMins + potentialMinsSize, h) != potentialMins + potentialMinsSize) {
					Range range;
					range.setJSize = 0;
					range.setJ[range.setJSize++] = h;

					if (yMax[h] == minmax) {
						range.yMin = yMax[h];
						range.yMax = yMax[h];
						yTmp = yMin[h];
						yMin[h] = yMax[h];
						r2 = calculateR2(h, range);
						yMin[h] = yTmp;
						result -= r2;
					}

					range.yMin = yMin[h];
					range.yMax = yMin[h];
					yTmp = yMax[h];
					yMax[h] = yMin[h];
					r2 = calculateR2(h, range);
					yMax[h] = yTmp;
					result += r2;
				}
			}
		}



		return result;
	}


	DType QR2Solver::QR2contribution(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset, R2Result* result, bool calculateHV) {
		if (end < start) {
			return 0;
		}

		NumberOfSubproblems++;
		offset++;
		offset = offset % currentSettings->NumberOfObjectives;

		int oldmaxIndexUsed = maxIndexUsed;

		// if there is just one point
		if (end == start) {
			if(calculateHV) result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[start]), &idealPoint);


			return calculateR2Contribution(nadirPoint, *(currentlySolvedProblem->points[start]), idealPoint, calculateHV);
		}

		// if there are just two points
		if (end - start == 1) {
			if (calculateHV) {
				result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[start]), &idealPoint);
				result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[end]), &idealPoint);
			}

			DType r2Contribution = calculateR2Contribution(nadirPoint, *(currentlySolvedProblem->points[start]), idealPoint, calculateHV);
			r2Contribution += calculateR2Contribution(nadirPoint, *(currentlySolvedProblem->points[end]), idealPoint, calculateHV);

			tmpPoint = *currentlySolvedProblem->points[start];
			for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
				tmpPoint.ObjectiveValues[j] = std::min(tmpPoint.ObjectiveValues[j], currentlySolvedProblem->points[end]->ObjectiveValues[j]);
			}

		if(calculateHV) result->Hypervolume -= Volume2(&nadirPoint, &tmpPoint, &idealPoint);
	

			r2Contribution -= calculateR2Contribution(nadirPoint, tmpPoint, idealPoint, calculateHV);
			return r2Contribution;
		}

		unsigned iPivot = start;
		DType maxVolume = Volume2(&nadirPoint, (currentlySolvedProblem->points)[iPivot], &idealPoint);

		// find the pivot point
		for (unsigned i = start + 1; i <= end; i++) {
			DType volumeCurrent = Volume2(&nadirPoint, (currentlySolvedProblem->points)[i], &idealPoint);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}


		if(calculateHV)result->Hypervolume += maxVolume;


		DType r2Contribution = calculateR2Contribution(nadirPoint, *(currentlySolvedProblem->points)[iPivot], idealPoint, calculateHV);

		// build subproblems
		unsigned iPos = maxIndexUsed + 1;
		short j, jj;

		Point partNadirPoint = nadirPoint;
		Point partIdealPoint = idealPoint;

		for (jj = 0; jj < currentSettings->NumberOfObjectives; jj++) {
			j = backend::off(offset, jj, currentSettings->NumberOfObjectives);

			if (jj > 0) {
				short j2 = backend::off(offset, jj - 1, currentSettings->NumberOfObjectives);
				partIdealPoint.ObjectiveValues[j2] = std::min(idealPoint.ObjectiveValues[j2], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = nadirPoint.ObjectiveValues[j2];
			}

			unsigned partStart = iPos;

			for (unsigned i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[i]->ObjectiveValues[j]) >
					std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j])) {
					currentlySolvedProblem->points[iPos++] = currentlySolvedProblem->points[i];
				}
			}

			unsigned partEnd = iPos - 1;
			maxIndexUsed = iPos - 1;

			if (partEnd >= partStart) {
				partNadirPoint.ObjectiveValues[j] = std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j]);
				r2Contribution += QR2contribution(partStart, partEnd, partIdealPoint, partNadirPoint, offset, result, calculateHV);
			}
		}

		maxIndexUsed = oldmaxIndexUsed;
		return r2Contribution;
	}


	void QR2Solver::QR2init(int start, int end, Point& idealPoint, Point& nadirPoint, unsigned offset, R2Result* result, bool calculateHV) {
		if (end < start) {
			return;
		}

		NumberOfSubproblems++;
		offset++;
		offset = offset % currentSettings->NumberOfObjectives;

		int oldmaxIndexUsed = maxIndexUsed;

		// if there is just one point
		if (end == start) {
			if(calculateHV)
			result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[start]), &idealPoint);


			result->R2 = calculateR2Tentative(nadirPoint, *(currentlySolvedProblem->points[start]), idealPoint, calculateHV);
		}

		// if there are just two points
		if (end - start == 1) {
			if (calculateHV)
			{
				result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[start]), &idealPoint);
				result->Hypervolume += Volume2(&nadirPoint, (currentlySolvedProblem->points[end]), &idealPoint);
			}

			result->R2 = calculateR2Tentative(nadirPoint, *(currentlySolvedProblem->points[start]), idealPoint, calculateHV);
			result->R2 += calculateR2Tentative(nadirPoint, *(currentlySolvedProblem->points[end]), idealPoint, calculateHV);
			tmpPoint = *currentlySolvedProblem->points[start];
			for (short j = 0; j < currentSettings->NumberOfObjectives; j++) {
				tmpPoint.ObjectiveValues[j] = std::min(tmpPoint.ObjectiveValues[j], currentlySolvedProblem->points[end]->ObjectiveValues[j]);
			}

			if(calculateHV) result->Hypervolume -= Volume2(&nadirPoint, &tmpPoint, &idealPoint);
	

			result->R2 -= calculateR2Tentative(nadirPoint, tmpPoint, idealPoint, calculateHV);
			return;
		}

		unsigned iPivot = start;
		DType maxVolume = Volume2(&nadirPoint, (currentlySolvedProblem->points)[iPivot], &idealPoint);

		// find the pivot point
		for (unsigned i = start + 1; i <= end; i++) {
			DType volumeCurrent = Volume2(&nadirPoint, (currentlySolvedProblem->points)[i], &idealPoint);
			if (maxVolume < volumeCurrent) {
				maxVolume = volumeCurrent;
				iPivot = i;
			}
		}

		//cout << "   >>> Pivot=" << maxVolume << " (" << iPivot << ")\n";

		if(calculateHV)
		result->Hypervolume += maxVolume;

		DType totalR2Contribution = calculateR2Tentative(nadirPoint, *(currentlySolvedProblem->points)[iPivot], idealPoint, calculateHV);

		// build subproblems
		unsigned iPos = maxIndexUsed + 1;
		short j, jj;

		Point partNadirPoint = nadirPoint;
		Point partIdealPoint = idealPoint;

		for (jj = 0; jj < currentSettings->NumberOfObjectives; jj++) {
			j = backend::off(offset, jj, currentSettings->NumberOfObjectives);

			if (jj > 0) {
				short j2 = backend::off(offset, jj - 1, currentSettings->NumberOfObjectives);
				partIdealPoint.ObjectiveValues[j2] = std::min(idealPoint.ObjectiveValues[j2], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j2]);
				partNadirPoint.ObjectiveValues[j2] = nadirPoint.ObjectiveValues[j2];
			}

			unsigned partStart = iPos;

			for (unsigned i = start; i <= end; i++) {
				if (i == iPivot)
					continue;

				if (std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[i]->ObjectiveValues[j]) >
					std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j])) {
					currentlySolvedProblem->points[iPos++] = currentlySolvedProblem->points[i];
				}
			}

			unsigned partEnd = iPos - 1;
			maxIndexUsed = iPos - 1;

			if (partEnd >= partStart) {
				partNadirPoint.ObjectiveValues[j] = std::min(idealPoint.ObjectiveValues[j], currentlySolvedProblem->points[iPivot]->ObjectiveValues[j]);
				totalR2Contribution += QR2contribution(partStart, partEnd, partIdealPoint, partNadirPoint, offset, result, calculateHV);
			}
		}

		maxIndexUsed = oldmaxIndexUsed;
		result->R2 = totalR2Contribution;

	}


	// Entry point of QR2Solver algorithm
	R2Result* QR2Solver::solveQR2(std::vector <Point*>& points, Point& idealPoint, Point& nadirPoint, bool calculateHV) {
		short numberOfPoints = points.size();
		NumberOfSubproblems = 0;



		// allocate memory
		currentlySolvedProblem->points.resize(20000000);

		for (int i = 0; i < numberOfPoints; i++) {
			currentlySolvedProblem->points[i] = points[i];
		}

		maxIndexUsed = numberOfPoints - 1;
		R2Result* result = new R2Result();

		if(calculateHV)
			result->Hypervolume = 0;
		

		QR2init(0, numberOfPoints - 1, idealPoint, nadirPoint, 0, result, calculateHV);



		// release memory
		//indexSetVec.clear();
		//indexSetVec.shrink_to_fit();

		//cout << "> MaxIndexUsed:\t  " << maxIndexUsed << endl;
		//cout << "> Subproblems:\t  " << 	NumberOfSubproblems << endl;
		
		
		return result;
	}

}