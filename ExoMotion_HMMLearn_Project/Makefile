DOCTEST_CMD = python2 -m doctest
DOCTEST_FILES = Motion_Datareader_Train_Test.py HMMExo_Train_Test.py TrainingStatistics.py \
 util.py \
 tests/test_training_statistics.py tests/utils.py tests/test_motion_datareader_train_test.py
PYTEST_CMD = python2 -m pytest

test: pytest doctest

doctest:
	@echo RUNNING UNITTESTS VIA DOCTEST...
	$(DOCTEST_CMD) $(DOCTEST_FILES)
	@echo ... DONE

pytest:
	@echo RUNNING UNITTESTS VIA PYTEST...
	$(PYTEST_CMD) --ignore=lib --ignore=src
	@echo ... DONE

clean:
	rm -f *.pyc
	rm -rf __pycache__
