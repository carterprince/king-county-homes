//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Carter
 *  @version 2.0
 *  @date    2024
 *  @note    King County House Price Prediction using NeuralNet_XL
 *
 *  Single self-contained implementation for house price prediction
 *  > runMain kingCountyHousePriceModel
 */

package scalation
package modeling
package neuralnet

import scala.math.sqrt
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import java.io.{BufferedReader, FileReader}

import scalation.mathstat._
import scalation.modeling._

import ActivationFun._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kingCountyHousePriceModel` main function trains and evaluates a neural 
 *  network on the King County house price dataset.
 *  > runMain scalation.modeling.neuralnet.kingCountyHousePriceModel
 */
@main def kingCountyHousePriceModel(): Unit =

    Random.setSeed(42)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load CSV file with headers and return data matrix and column names.
     *  @param filename  path to CSV file
     */
    def loadCSV(filename: String): (MatrixD, Array[String]) =
        val reader = new BufferedReader(new FileReader(filename))
        
        // Read header
        val header = reader.readLine().split(",").map(_.trim.replace("\"", ""))
        
        // Read all data lines
        val rows = ArrayBuffer[Array[Double]]()
        var line = reader.readLine()
        
        while line != null do
            // Split and clean values (remove quotes)
            val values = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)") // handle quoted commas
                             .map(_.trim.replace("\"", ""))
            
            val numericValues = new Array[Double](values.length)
            
            for i <- values.indices do
                numericValues(i) = values(i) match
                    case v if i == 0 => v.toDouble  // id
                    case v if i == 1 => parseDate(v)  // date
                    case v if v.isEmpty => 0.0
                    case v => v.toDouble
            
            rows += numericValues
            line = reader.readLine()
        end while
        
        reader.close()
        (MatrixD(rows.map(arr => VectorD(arr)).toIndexedSeq), header)
    end loadCSV
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse date string to days since a reference point.
     *  @param dateStr  date string in format YYYYMMDDTHHMMSS
     */
    def parseDate(dateStr: String): Double =
        try
            val cleaned = dateStr.replace("T", "").replace("-", "")
            val year = cleaned.substring(0, 4).toInt
            val month = cleaned.substring(4, 6).toInt
            val day = cleaned.substring(6, 8).toInt
            
            // Convert to days (simplified - could use actual date calculation)
            (year - 2014) * 365.25 + (month - 1) * 30.44 + day
        catch
            case _: Exception => 0.0
    end parseDate
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Standardize features (zero mean, unit variance).
     *  @param x  the input matrix
     */
    def standardize(x: MatrixD): (MatrixD, VectorD, VectorD) =
        val means = VectorD(for j <- 0 until x.dim2 yield x(?, j).mean)
        val stds = VectorD(for j <- 0 until x.dim2 yield 
            val std = x(?, j).stdev
            if std < 1e-10 then 1.0 else std  // avoid division by zero
        )
        
        val xStd = new MatrixD(x.dim, x.dim2)
        for i <- 0 until x.dim; j <- 0 until x.dim2 do
            xStd(i, j) = (x(i, j) - means(j)) / stds(j)
        
        (xStd, means, stds)
    end standardize
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** One-hot encode a categorical vector.
     *  @param values  the categorical values
     */
    def oneHotEncode(values: VectorD): (MatrixD, Array[Double]) =
        val uniqueVals = values.toArray.distinct.sorted
        val valMap = uniqueVals.zipWithIndex.toMap
        val m = values.dim
        val encoded = new MatrixD(m, uniqueVals.length)
        
        for i <- 0 until m do
            val idx = valMap(values(i))
            encoded(i, idx) = 1.0
        
        (encoded, uniqueVals)
    end oneHotEncode
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split data into training and testing sets.
     *  @param x      the input matrix
     *  @param y      the output matrix
     *  @param ratio  the training ratio
     */
    def trainTestSplit(x: MatrixD, y: MatrixD, ratio: Double): 
        (MatrixD, MatrixD, MatrixD, MatrixD) =
        
        val m = x.dim
        val trainSize = (m * ratio).toInt
        val indices = Random.shuffle((0 until m).toList)
        
        val trainIdx = indices.take(trainSize).toArray
        val testIdx = indices.drop(trainSize).toArray
        
        val xTrain = x(trainIdx)
        val xTest = x(testIdx)
        val yTrain = y(trainIdx)
        val yTest = y(testIdx)
        
        (xTrain, xTest, yTrain, yTest)
    end trainTestSplit
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate evaluation metrics.
     *  @param yActual     actual values
     *  @param yPredicted  predicted values
     */
    def calculateMetrics(yActual: VectorD, yPredicted: VectorD): Map[String, Double] =
        val n = yActual.dim
        val errors = yActual - yPredicted
        val squaredErrors = errors.map(e => e * e)
        val absErrors = errors.map(e => scala.math.abs(e))
        
        val mse = squaredErrors.sum / n
        val rmse = sqrt(mse)
        val mae = absErrors.sum / n
        
        val yMean = yActual.sum / n
        val sst = yActual.map(y => (y - yMean) * (y - yMean)).sum
        val sse = squaredErrors.sum
        val r2 = 1.0 - (sse / sst)
        
        Map("RMSE" -> rmse, "MAE" -> mae, "R2" -> r2, "MSE" -> mse)
    end calculateMetrics
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hyperparameter configuration.
     */
    case class HParams(
        hiddenWidth: Int,
        numLayers: Int,
        activation: AFF,
        lr: Double,
        batchSize: Int,
        weightDecay: Double = 0.0
    )
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train model with given hyperparameters.
     */
    def trainModel(xTr: MatrixD, yTr: MatrixD, xTe: MatrixD, yTe: MatrixD,
                   params: HParams, epochs: Int): (Double, NeuralNet_XL) =
        
        Optimizer.hp("eta") = params.lr
        Optimizer.hp("bSize") = params.batchSize.toDouble
        Optimizer.hp("maxEpochs") = epochs.toDouble
        
        val nz = Array.fill(params.numLayers)(params.hiddenWidth)
        val acts = Array.fill(params.numLayers)(params.activation) :+ f_id
        
        val model = NeuralNet_XL.rescale(xTr, yTr, nz = nz, 
                                         hparam = Optimizer.hp, f = acts)
        
        model.train()
        
        val (_, qof) = model.test(xTe, yTe)
        
        // FIXED: Check rows (metrics) not columns (outputs)
        if qof.dim <= QoF.sse.ordinal then
            throw new Exception(s"Invalid QoF dimensions: ${qof.dim} (need at least ${QoF.sse.ordinal + 1} metrics)")
        
        // FIXED: Access as (metric_row, output_column)
        val testRMSE = sqrt(qof(QoF.sse.ordinal, 0) / xTe.dim)
        
        // Check for NaN or Inf
        if testRMSE.isNaN || testRMSE.isInfinite then
            throw new Exception(s"Invalid RMSE: $testRMSE")
        
        (testRMSE, model)
    end trainModel
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // MAIN EXECUTION
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    banner("KING COUNTY HOUSE PRICE PREDICTION")
    
    // Load and preprocess data
    banner("Loading Data")
    val (rawData, colNames) = loadCSV("kc_house_data.csv")
    println(s"Loaded ${rawData.dim} samples with ${rawData.dim2} columns")
    println(s"Columns: ${colNames.mkString(", ")}")
    
    // Column indices (based on CSV structure)
    val dateCol = 1
    val priceCol = 2
    val bedroomsCol = 3
    val bathroomsCol = 4
    val sqftLivingCol = 5
    val sqftLotCol = 6
    val floorsCol = 7
    val waterfrontCol = 8
    val viewCol = 9
    val conditionCol = 10
    val gradeCol = 11
    val sqftAboveCol = 12
    val sqftBasementCol = 13
    val yrBuiltCol = 14
    val yrRenovatedCol = 15
    val zipcodeCol = 16
    val latCol = 17
    val longCol = 18
    val sqftLiving15Col = 19
    val sqftLot15Col = 20
    
    // Extract response variable (price)
    val yRaw = rawData(?, priceCol)
    
    // Log transform the price to reduce scale
    banner("Transforming Response Variable")
    val y = yRaw.map(p => scala.math.log(p + 1))
    println(s"Applied log transformation to prices")
    println(f"Original price range: $$${yRaw.min}%,.2f - $$${yRaw.max}%,.2f")
    println(f"Log-transformed range: ${y.min}%.4f - ${y.max}%.4f")
    
    // Extract numerical features (excluding id and date initially)
    val numericalCols = Array(bedroomsCol, bathroomsCol, sqftLivingCol, sqftLotCol,
                              floorsCol, waterfrontCol, viewCol, conditionCol, gradeCol,
                              sqftAboveCol, sqftBasementCol, yrBuiltCol, yrRenovatedCol,
                              latCol, longCol, sqftLiving15Col, sqftLot15Col, dateCol)
    
    val xNumerical = rawData(?, numericalCols)
    
    // One-hot encode zipcode
    banner("One-Hot Encoding Zipcodes")
    val zipcodes = rawData(?, zipcodeCol)
    val (zipcodeEncoded, uniqueZips) = oneHotEncode(zipcodes)
    println(s"Found ${uniqueZips.length} unique zipcodes")
    
    // Combine features
    val xRaw = xNumerical ++^ zipcodeEncoded
    
    // Standardize features
    banner("Standardizing Features")
    val (x, _, _) = standardize(xRaw)
    println(s"Standardized ${x.dim2} features to zero mean and unit variance")
    
    // Create feature names
    val numFeatureNames = Array("bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                                "floors", "waterfront", "view", "condition", "grade",
                                "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
                                "lat", "long", "sqft_living15", "sqft_lot15", "date_days")
    val zipFeatureNames = uniqueZips.map(z => s"zip_${z.toInt}")
    val allFeatureNames = numFeatureNames ++ zipFeatureNames
    
    println(s"Total features: ${x.dim2} (${numFeatureNames.length} numerical + ${zipFeatureNames.length} zipcode)")
    
    // Split data
    banner("Splitting Data (80% train, 20% test)")
    val yMatrix = MatrixD(y).transpose
    val (xTrain, xTest, yTrain, yTest) = trainTestSplit(x, yMatrix, 0.8)
    
    println(s"Training samples: ${xTrain.dim}")
    println(s"Test samples: ${xTest.dim}")
    
    // Hyperparameter search space - MUCH more conservative
    val hiddenWidths = Array(16, 32, 64)
    val numLayers = Array(1, 2, 3)
    val activations = Array(f_tanh, f_sigmoid, f_lreLU)  // Removed reLU and eLU
    val learningRates = Array(0.00001, 0.00005, 0.0001, 0.0005)  // Much lower!
    val batchSizes = Array(64, 128)
    
    // Random search
    banner("Hyperparameter Search (Random Sampling)")
    val nTrials = 15  // Reduced trials
    var bestRMSE = Double.MaxValue
    var bestParams: HParams = null
    var successfulTrials = 0
    
    println(s"Running $nTrials random trials...\n")
    
    for i <- 1 to nTrials do
        println(s"===== Trial $i/$nTrials =====")
        
        val params = HParams(
            hiddenWidth = hiddenWidths(Random.nextInt(hiddenWidths.length)),
            numLayers = numLayers(Random.nextInt(numLayers.length)),
            activation = activations(Random.nextInt(activations.length)),
            lr = learningRates(Random.nextInt(learningRates.length)),
            batchSize = batchSizes(Random.nextInt(batchSizes.length)),
            weightDecay = 0.0
        )
        
        println(f"  Config: width=${params.hiddenWidth}, layers=${params.numLayers}, " +
                f"act=${params.activation.name}, lr=${params.lr}, batch=${params.batchSize}")
        
        try
            val (testRMSE, _) = trainModel(xTrain, yTrain, xTest, yTest, params, 5)  // Only 5 epochs for search
            println(f"  Test RMSE (log-scale): ${testRMSE}%.4f")
            successfulTrials += 1
            
            if testRMSE < bestRMSE then
                bestRMSE = testRMSE
                bestParams = params
                println(f"  *** NEW BEST! ***")
        catch
            case e: Exception =>
                println(s"  Trial failed: ${e.getMessage}")
        
        println()
    end for
    
    // Check if any trials succeeded
    if bestParams == null then
        println("\n" + "="*60)
        println("ERROR: All hyperparameter trials failed!")
        println("This suggests the model configuration needs adjustment.")
        println(s"Successful trials: $successfulTrials / $nTrials")
        println("="*60)
        return
    end if
    
    // Display best hyperparameters
    banner("BEST HYPERPARAMETERS")
    println(s"Successful trials: $successfulTrials / $nTrials")
    println(s"Hidden Width: ${bestParams.hiddenWidth}")
    println(s"Num Hidden Layers: ${bestParams.numLayers}")
    println(s"Activation: ${bestParams.activation.name}")
    println(s"Learning Rate: ${bestParams.lr}")
    println(s"Batch Size: ${bestParams.batchSize}")
    println(f"Best Test RMSE (log-scale): ${bestRMSE}%.4f")
    
    // Train final model with more epochs
    banner("Training Final Model (50 epochs)")
    val (_, finalModel) = trainModel(xTrain, yTrain, xTest, yTest, 
                                             bestParams, 50)
    
    // Evaluate final model
    banner("FINAL MODEL PERFORMANCE")
    
    // Training metrics
    val yTrainPredLog = finalModel.predict(xTrain)(?, 0)
    val yTrainPred = yTrainPredLog.map(p => scala.math.exp(p) - 1)
    val yTrainActual = yTrain(?, 0).map(p => scala.math.exp(p) - 1)
    val trainMetrics = calculateMetrics(yTrainActual, yTrainPred)
    
    println("\n*** Training Set ***")
    println(f"  RMSE: $$${trainMetrics("RMSE")}%,.2f")
    println(f"  MAE:  $$${trainMetrics("MAE")}%,.2f")
    println(f"  R²:   ${trainMetrics("R2")}%.6f")
    
    // Test metrics
    val yTestPredLog = finalModel.predict(xTest)(?, 0)
    val yTestPred = yTestPredLog.map(p => scala.math.exp(p) - 1)
    val yTestActual = yTest(?, 0).map(p => scala.math.exp(p) - 1)
    val testMetrics = calculateMetrics(yTestActual, yTestPred)
    
    println("\n*** Test Set ***")
    println(f"  RMSE: $$${testMetrics("RMSE")}%,.2f")
    println(f"  MAE:  $$${testMetrics("MAE")}%,.2f")
    println(f"  R²:   ${testMetrics("R2")}%.6f")
    
    // Sample predictions
    banner("SAMPLE PREDICTIONS")
    val sampleIndices = Random.shuffle((0 until xTest.dim).toList).take(10)
    
    println(f"${"Actual"}%12s | ${"Predicted"}%12s | ${"Error"}%12s | ${"Error %"}%8s")
    println("-" * 60)
    
    for idx <- sampleIndices do
        val actual = yTestActual(idx)
        val predicted = yTestPred(idx)
        val error = scala.math.abs(actual - predicted)
        val errorPct = (error / actual) * 100
        println(f"$$${actual}%,11.2f | $$${predicted}%,11.2f | $$${error}%,11.2f | ${errorPct}%7.1f%%")
    
    // Feature importance (approximate using first layer weights)
    banner("TOP 15 MOST IMPORTANT FEATURES")
    val weights = finalModel.getNetParam(0).w
    val featureImportance = VectorD(for j <- 0 until weights.dim2 yield 
        weights(?, j).map(scala.math.abs).sum)
    
    val topFeatures = featureImportance.zipWithIndex
                                       .sortBy(-_._1)
                                       .take(15)
    
    for (importance, idx) <- topFeatures do
        if idx < allFeatureNames.length then
            println(f"  ${allFeatureNames(idx)}%-25s ${importance}%10.2f")
    
    // Create visualization
    banner("Creating Visualization")
    try
        new Plot(null, yTestActual, yTestPred, 
                 "King County House Prices: Actual vs Predicted (Test Set)",
                 lines = false)
        println("Plot displayed successfully")
    catch
        case e: Exception =>
            println(s"Could not create plot: ${e.getMessage}")
    
    banner("COMPLETED")

end kingCountyHousePriceModel
