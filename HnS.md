Below is an example of a set of pseudocode algorithms that follow the outlined steps to detect a
Head & Shoulders (H&S) pattern. You can adapt these to your preferred programming language.
1. Identify Peaks (Left Shoulder, Head, Right Shoulder)
function findPeaks(prices):
peaks = [] // list of indices where a peak is detected
n = length(prices)
for i from 1 to n-2:
// A simple local maximum: greater than its immediate neighbors
if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
peaks.append(i)
return peaks
Explanation:
• The function findPeaks scans through the price series.
• It detects a local maximum when the price is greater than the prices immediately before and
after.
• The resulting list of indices is used for the next steps.
2. Identify the H&S Structure
function identifyHS(prices, peaks, shoulderTolerance):
// shoulderTolerance is a relative difference allowed between the left and right shoulders
hsPatterns = [] // to store valid H&S patterns: each pattern is a tuple (LS, H, RS)
// We need at least three peaks: left shoulder, head, and right shoulder.
for i from 0 to length(peaks) - 3:
LS = peaks[i]
H = peaks[i+1]
RS = peaks[i+2]
// Conditions for a valid pattern:
// Head must be higher than both shoulders.
if prices[H] > prices[LS] and prices[H] > prices[RS]:
// Shoulders should be roughly equal (within a tolerance)
shoulderDiff = abs(prices[LS] - prices[RS]) / max(prices[LS], prices[RS])
if shoulderDiff <= shoulderTolerance:
hsPatterns.append((LS, H, RS))
return hsPatterns
Explanation:
• This function takes the list of peaks and looks for three consecutive peaks where:
o The middle peak (head) is the highest.
o The left shoulder (LS) and right shoulder (RS) are of similar height (within a given
tolerance).
• Valid tuples of indices are stored for further analysis.
3. Draw the Neckline and Confirm the Break
function drawNeckline(prices, LS, H, RS):
// Identify troughs (local minima) between LS-H and H-RS.
// For simplicity, search between the index ranges.
T1 = findTrough(prices, start=LS, end=H)
T2 = findTrough(prices, start=H, end=RS)
// The neckline is a line connecting (T1, prices[T1]) and (T2, prices[T2])
return (T1, T2)
function findTrough(prices, start, end):
// Find index of minimum price between start and end indices.
trough = start
for i from start+1 to end:
if prices[i] < prices[trough]:
trough = i
return trough
function confirmNecklineBreak(prices, neckline, RS, confirmationLookahead):
(T1, T2) = neckline
// Calculate the neckline price at a given x (for simplicity, use linear interpolation)
// Here, we consider the neckline as the line from T1 to T2.
slope = (prices[T2] - prices[T1]) / (T2 - T1)
// Look for a break in the price after RS over a confirmation period
for i from RS+1 to RS+confirmationLookahead:
// Calculate the expected neckline price at i:
necklinePrice = prices[T1] + slope * (i - T1)
if prices[i] < necklinePrice:
// A confirmed break (for a bearish H&S, price breaks below the neckline)
return true
return false
Explanation:
• drawNeckline:
o Finds the lowest point (trough) between the left shoulder and head (T1) and
between head and right shoulder (T2).
o Connects these two troughs to form the neckline.
• confirmNecklineBreak:
o Checks if, after the right shoulder, the price breaks below the neckline over a
specified lookahead period.
o Uses linear interpolation to determine the neckline price at each point.
4. Analyze Volume Trends
function analyzeVolume(volumeSeries, LS, H, RS, postBreakIndex):
// Example criteria:
// - Volume is highest during the head formation.
// - Volume on the right shoulder is lower than during the head.
// - Volume increases again on the break of the neckline.
headVolume = volumeSeries[H]
leftVolume = volumeSeries[LS]
rightVolume = volumeSeries[RS]
breakVolume = volumeSeries[postBreakIndex] // volume at the confirmed break index
criteria1 = headVolume > leftVolume and headVolume > rightVolume
criteria2 = rightVolume < headVolume
criteria3 = breakVolume > volumeSeries[RS] // volume surge at break
if criteria1 and criteria2 and criteria3:
return true
else:
return false
Explanation:
• This function compares volume at key points:
o The head should show higher volume compared to the shoulders.
o The volume on the right shoulder should be lower.
o There should be an increase in volume when the neckline is broken.
• These conditions help validate the pattern from a volume perspective.
5. Calculate the Price Target and Validate the Outcome
function calculatePriceTarget(prices, neckline, H, patternType="bearish"):
(T1, T2) = neckline
// Determine the vertical distance (pattern height) from the head to the neckline at H.
// Interpolate neckline price at H:
slope = (prices[T2] - prices[T1]) / (T2 - T1)
necklineAtH = prices[T1] + slope * (H - T1)
patternHeight = prices[H] - necklineAtH
if patternType == "bearish":
// For a bearish H&S, price target is measured downward from the neckline break point.
target = necklineAtH - patternHeight
else:
// For an inverse H&S (bullish reversal), measured upward.
target = necklineAtH + patternHeight
return target
Explanation:
• The price target is calculated by:
o Measuring the distance (vertical difference) between the head and the neckline.
o Projecting that distance from the breakout point.
o For a standard (bearish) H&S, the target is below the neckline.
6. Bringing It All Together
function detectHSPattern(prices, volumeSeries, shoulderTolerance, confirmationLookahead):
peaks = findPeaks(prices)
hsCandidates = identifyHS(prices, peaks, shoulderTolerance)
for each candidate in hsCandidates:
(LS, H, RS) = candidate
neckline = drawNeckline(prices, LS, H, RS)
if confirmNecklineBreak(prices, neckline, RS, confirmationLookahead):
// Assume break is confirmed at index 'breakIndex' (could be captured within the confirmation
function)
breakIndex = RS + 1 // Simplification: use the first index after RS where break occurred
// Check volume criteria
if analyzeVolume(volumeSeries, LS, H, RS, breakIndex):
target = calculatePriceTarget(prices, neckline, H, "bearish")
// Validate outcome (e.g., further criteria or risk management can be added here)
return {
"pattern": "Head & Shoulders",
"left_shoulder_index": LS,
"head_index": H,
"right_shoulder_index": RS,
"neckline": neckline,
"break_index": breakIndex,
"price_target": target
}
// No valid pattern found
return null
Explanation:
• The detectHSPattern function brings together all the previous steps:
1. Find Peaks: Identify local peaks in the price series.
2. Identify H&S Candidates: Look for three consecutive peaks that meet the criteria.
3. Draw and Confirm Neckline: Calculate the neckline and check if the price breaks
below it.
4. Analyze Volume: Validate the pattern using volume trends.
5. Calculate Price Target: Compute the target price based on the pattern’s height.
• If all conditions are met, the function returns details of the detected pattern.
Final Notes
• Thresholds & Tolerances:
Adjust shoulderTolerance (e.g., 10%-20%) and confirmationLookahead based on your
market/timeframe.
• Enhancements:
In a production algorithm, you might include additional filters (e.g., time constraints between
peaks, smoothing of price data, more advanced breakout confirmation) to reduce false
positives.
This set of pseudocode algorithms outlines one method to detect and validate Head & Shoulders
patterns based on price and volume. You can translate and refine this pseudocode in your chosen
programming language.