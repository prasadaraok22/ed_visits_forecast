# ed_visits_forecast
# Tech Stack
-- Python
-- TensorFlow
-- scikit-learn
-- FastAPI
-- uvicorn
--


# Cretae virtual environment    
<ol class="linenums"><li class="L0"><code class="language-shell"><span class="pln">pip install virtualenv  # OR pip3 install virtualenv </span></code></li><li class="L1"><code class="language-shell"><span class="pln">virtualenv my_env </span><span class="com"># create a virtual environment named my_env</span></code></li><li class="L2"><code class="language-shell"><span class="pln">source my_env</span><span class="pun">/</span><span class="pln">bin</span><span class="pun">/</span><span class="pln">activate </span><span class="com"># activate my_env</span></code></li></ol>


# Install necessary packages
pip install -r requirements.txt

# Run the below file once to train and save the model (Note: Run only once)
python train_and_save_model.py.py

# Run the below file to test the prediction quickly (Optional step)
python test_model.py


