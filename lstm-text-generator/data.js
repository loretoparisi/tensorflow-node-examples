/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018-2019 Loreto Parisi (loretoparisi@gmail.com)
 */

 
const tf = require('@tensorflow/tfjs-node');

class TextData {
    /**
     * Constructor of TextData.
     *
     * @param {string} dataIdentifier An identifier for this instance of TextData.
     * @param {string} textString The training text data.
     * @param {number} sampleLen Length of each training example, i.e., the input
     *   sequence length expected by the LSTM model.
     * @param {number} sampleStep How many characters to skip when going from one
     *   example of the training data (in `textString`) to the next.
     */
    constructor(dataIdentifier, textString, sampleLen, sampleStep) {
      tf.util.assert(
          sampleLen > 0,
          `Expected sampleLen to be a positive integer, but got ${sampleLen}`);
      tf.util.assert(
          sampleStep > 0,
          `Expected sampleStep to be a positive integer, but got ${sampleStep}`);
  
      if (!dataIdentifier) {
        throw new Error('Model identifier is not provided.');
      }
  
      this.dataIdentifier_ = dataIdentifier;
  
      this.textString_ = textString;
      this.textLen_ = textString.length;
      this.sampleLen_ = sampleLen;
      this.sampleStep_ = sampleStep;
  
      this.getCharSet_();
      this.convertAllTextToIndices_();
    }
  
    /**
     * Get data identifier.
     *
     * @returns {string} The data identifier.
     */
    dataIdentifier() {
      return this.dataIdentifier_;
    }
  
    /**
     * Get length of the training text data.
     *
     * @returns {number} Length of training text data.
     */
    textLen() {
      return this.textLen_;
    }
  
    /**
     * Get the length of each training example.
     */
    sampleLen() {
      return this.sampleLen_;
    }
  
    /**
     * Get the size of the character set.
     *
     * @returns {number} Size of the character set, i.e., how many unique
     *   characters there are in the training text data.
     */
    charSetSize() {
      return this.charSetSize_;
    }
  
    /**
     * Generate the next epoch of data for training models.
     *
     * @param {number} numExamples Number examples to generate.
     * @returns {[tf.Tensor, tf.Tensor]} `xs` and `ys` Tensors.
     *   `xs` has the shape of `[numExamples, this.sampleLen, this.charSetSize]`.
     *   `ys` has the shape of `[numExamples, this.charSetSize]`.
     */
    nextDataEpoch(numExamples) {
      this.generateExampleBeginIndices_();
  
      if (numExamples == null) {
        numExamples = this.exampleBeginIndices_.length;
      }
  
      const xsBuffer = new tf.TensorBuffer([
          numExamples, this.sampleLen_, this.charSetSize_]);
      const ysBuffer  = new tf.TensorBuffer([numExamples, this.charSetSize_]);
      for (let i = 0; i < numExamples; ++i) {
        const beginIndex = this.exampleBeginIndices_[
            this.examplePosition_ % this.exampleBeginIndices_.length];
        for (let j = 0; j < this.sampleLen_; ++j) {
          xsBuffer.set(1, i, j, this.indices_[beginIndex + j]);
        }
        ysBuffer.set(1, i, this.indices_[beginIndex + this.sampleLen_]);
        this.examplePosition_++;
      }
      return [xsBuffer.toTensor(), ysBuffer.toTensor()];
    }
  
    /**
     * Get the unique character at given index from the character set.
     *
     * @param {number} index
     * @returns {string} The unique character at `index` of the character set.
     */
    getFromCharSet(index) {
      return this.charSet_[index];
    }
  
    /**
     * Convert text string to integer indices.
     *
     * @param {string} text Input text.
     * @returns {number[]} Indices of the characters of `text`.
     */
    textToIndices(text) {
      const indices = [];
      for (let i = 0; i < text.length; ++i) {
        indices.push(this.charSet_.indexOf(text[i]));
      }
      return indices;
    }
  
    /**
     * Get a random slice of text data.
     *
     * @returns {[string, number[]} The string and index representation of the
     *   same slice.
     */
    getRandomSlice() {
      const startIndex =
          Math.round(Math.random() * (this.textLen_ - this.sampleLen_ - 1));
      const textSlice = this.slice_(startIndex, startIndex + this.sampleLen_);
      return [textSlice, this.textToIndices(textSlice)];
    }
  
    /**
     * Get a slice of the training text data.
     *
     * @param {number} startIndex
     * @param {number} endIndex
     * @param {bool} useIndices Whether to return the indices instead of string.
     * @returns {string | Uint16Array} The result of the slicing.
     */
    slice_(startIndex, endIndex) {
      return this.textString_.slice(startIndex, endIndex);
    }
  
    /**
     * Get the set of unique characters from text.
     */
    getCharSet_() {
      this.charSet_ = [];
      for (let i = 0; i < this.textLen_; ++i) {
        if (this.charSet_.indexOf(this.textString_[i]) === -1) {
          this.charSet_.push(this.textString_[i]);
        }
      }
      this.charSetSize_ = this.charSet_.length;
    }
  
    /**
     * Convert all training text to integer indices.
     */
    convertAllTextToIndices_() {
      this.indices_ = new Uint16Array(this.textToIndices(this.textString_));
    }
  
    /**
     * Generate the example-begin indices; shuffle them randomly.
     */
    generateExampleBeginIndices_() {
      // Prepare beginning indices of examples.
      this.exampleBeginIndices_ = [];
      for (let i = 0;
          i < this.textLen_ - this.sampleLen_ - 1;
          i += this.sampleStep_) {
        this.exampleBeginIndices_.push(i);
      }
  
      // Randomly shuffle the beginning indices.
      tf.util.shuffle(this.exampleBeginIndices_);
      this.examplePosition_ = 0;
    }
  }

  module.exports = TextData;