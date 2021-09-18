#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def getClassStats(inputDict, className):

    outStats = {}

    for item in inputDict:

        val = item[className]
        if val not in outStats:
            outStats[val] = 0

        outStats[val] += 1

    return outStats


def buildDictStats(inputDict, classList):

    locStats = {"total": len(inputDict)}

    for cat in classList:

        locStats[cat] = getClassStats(inputDict, cat)

    return locStats


def buildKeyOrder(shiftAttrib,
                  shiftAttribVal,
                  stats=None):
    r"""
    If the dataset is labelled, give the order in which the attributes are given

    Args:

        - shiftAttrib (dict): order of each category in the category vector
        - shiftAttribVal (dict): list (ordered) of each possible labels for each
                                category of the category vector
        - stats (dict): if not None, number of representant of each label for
                        each category. Will update the output dictionary with a
                        "weights" index telling how each labels should be
                        balanced in the classification loss.

    Returns:

        A dictionary output[key] = { "order" : int , "values" : list of string}
    """

    MAX_VAL_EQUALIZATION = 10

    output = {}
    for key in shiftAttrib:
        output[key] = {}
        output[key]["order"] = shiftAttrib[key]
        output[key]["values"] = [None for i in range(len(shiftAttribVal[key]))]
        for cat, shift in shiftAttribVal[key].items():
            output[key]["values"][shift] = cat

    if stats is not None:
        for key in output:

            n = sum([x for key, x in stats[key].items()])

            output[key]["weights"] = {}
            for item, value in stats[key].items():
                output[key]["weights"][item] = min(
                    MAX_VAL_EQUALIZATION, n / float(value + 1.0))

    return output
