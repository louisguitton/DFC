package dfc;

import Util.Dictionary;
import Util.LuceneHandler;
import Util.MathUtil;
import Util.Pair;

import org.apache.lucene.document.Document;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by XJ on 2016/3/10. Update by Csq on 2017/4/7 Update the
 * function:load(),initCont()
 */
public class LoadDocs implements Decorator {

	private SModel model;
	private float soomth;
	public int seedwordnum;

	public LoadDocs(SModel model) {
		this.model = model;
		// soomth = (float) 1 / model.cateNum;
		soomth = (float) 1 / model.iCateNum;
		seedwordnum = 0;
	}

	/*
	 * �ж��ĵ�������ԣ���ʼ��ÿƪ�й��ĵ� param (�ĵ����ݣ��ĵ�����)
	 */
	private void initContForFilter(String cont, SDocument sDoc) {
		String[] ves = cont.split(" ");

		// ��ǰ�ĵ�������interesting category��seed word���� f(d,c),���ڼ����d
		double[] re = new double[model.iCateNum];
		// int count=0;
		for (String s : ves) {
			// ֻҪ����һ������seed word�ͰѸ��ĵ���ʼ��Ϊ�й��ĵ�
			for (int cate = 0; cate < model.iCateNum; cate++)
				if (model.seedSet[cate].contains(s)) {
					sDoc.y = 1;
					// count++;
					// seedwordnum++;
					re[cate]++;
				}
			/*
			 * if(count>4) sDoc.y=1;
			 */
		}
		// �����ƪ�ĵ��Ħ�d(c)��c��interesting category
		initEta(re, sDoc.index);

		// α����ĵ�
		if (sDoc.y == 1) {
			int cate = 0;
			// double maxEta = model.eta[sDoc.index][0];
			// ���ݦǽ������Ĳ���
			// cate = MathUtil.sample_neg(model.eta[sDoc.index]);
			cate = MathUtil.sample(model.eta[sDoc.index]);
			// �õ���ǰ�ĵ��ĳ�ʼ�����ȡ�����ֵ
			/*
			 * for(int i=1;i<model.iCateNum;i++){ if(model.eta[sDoc.index][i]>maxEta){
			 * maxEta=model.eta[sDoc.index][i]; cate=i; } }
			 */
			sDoc.prediction = cate;
			int word, topic;
			for (String s : ves) {
				word = Dictionary.contains(s);
				if (word == -1)
					continue;
				if (model.seedSet[sDoc.prediction].contains(Dictionary.getWord(word)) && Math.random() < 0.5
						|| Math.random() < (model.rho)) {
					// if
					// (model.seedSet[sDoc.prediction].contains(Dictionary.getWord(word))&&
					// Math.random() < 0.5|| Math.random() < 0.5) {
					// if(Math.random()<model.tao[word][sDoc.prediction]){
					sDoc.xvec.add(0);
					// docX[0]ͳ�Ƶ��ǵ�ǰ�ĵ���category word���� docX[1]ͳ�Ʒ�category word��
					model.docX[sDoc.index][0]++;
					// x=0������£�ÿ���ʵ�topic����ʼ��Ϊ-1
					sDoc.contents.add(new Pair(word, -1));
					// Ncw++,Ncw word���ֵ�category c�Ĵ���
					model.categoryWord[sDoc.prediction][word]++;
					// ��ǰ�ĵ���category Cd�µ�category word�� ��Ncw
					model.numCWords4Cate[sDoc.prediction]++;
				}
				// ����category word
				else {
					model.docX[sDoc.index][1]++;
					sDoc.xvec.add(1);
					// ���Ϊx=1��word ����һ��general-topic
					topic = (int) (Math.random() * model.topicNum);
					sDoc.contents.add(new Pair(word, topic));
					// Nct++��Nct�� word��cate c�µ��ĵ����ұ��ֵ�topic t��
					model.categoryTopic[sDoc.prediction][topic]++;
					// Ndt++,Ndt: �ĵ�d�еĴʱ��ֵ�topic t�µĴ���
					model.docTopic[sDoc.index][topic]++;
					// Ntw++,Ntw: word w���ֵ�general-topic t��
					model.wordTopic[word][topic]++;
					// ��Nct,�����ĵ������c���ĵ������д� �� x=1������category word (the total
					// number of words within the doc of category c and the
					// words are not category words)
					model.numRWords4Cate[sDoc.prediction]++;
					// ��Ntw,�ֵ�topic t�µ��ܴ���
					model.numRWords4Topic[topic]++;
				}

			}
			sDoc.doclength = sDoc.contents.size();
		} // end sDoc.y==1

	}

	// classification
	private void load() {
		LuceneHandler lh = new LuceneHandler(model.luceneIndexPath);
		Document[] lucDocs = lh.getDocs();

		int len = model.docNum;
		if (len == 0) {
			System.out.println("there is no document in the path");
			System.exit(-1);
		}
		SDocument sDoc;
		Document luceneDoc;

		for (int i = 0; i < len; i++) {
			model.documents[i] = new SDocument();
			luceneDoc = lucDocs[i];
			sDoc = model.documents[i];

			sDoc.index = i;
			model.kappa[i] = model.kd;
			sDoc.y = 1;
			sDoc.scores = new double[model.cateNum];
			sDoc.title = luceneDoc.get(LuceneHandler.TITLE);
			sDoc.groundTruth = Integer.parseInt(luceneDoc.get(LuceneHandler.CATE));
			sDoc.check = Integer.parseInt(luceneDoc.get(LuceneHandler.CHECK)) == 1 ? true : false;

			sDoc.xvec = new ArrayList<>();
			if (sDoc.check)
				model.groundTruth[sDoc.groundTruth]++;
			initCont(luceneDoc.get(LuceneHandler.ABSTRACT), sDoc);
		}
	}

	// classification
	private void initCont(String cont, SDocument sDoc) {

		String[] ves = cont.split(" ");
		double[] re = new double[model.cateNum];

		for (String s : ves)
			for (int cate = 0; cate < model.cateNum; cate++)
				if (model.seedSet[cate].contains(s))
					re[cate]++;
		initEta(re, sDoc.index);
		sDoc.prediction = MathUtil.sample(model.eta[sDoc.index]);
		int word, topic;

		for (String s : ves) {
			word = Dictionary.contains(s);
			if (word == -1)
				continue;
			// category word
			if (model.seedSet[sDoc.prediction].contains(Dictionary.getWord(word)) && Math.random() < 0.5
					|| Math.random() < (model.rho)) {
				sDoc.xvec.add(0);
				model.docX[sDoc.index][0]++;
				sDoc.contents.add(new Pair(word, -1));
				model.categoryWord[sDoc.prediction][word]++;
				model.numCWords4Cate[sDoc.prediction]++;
			}
			// nor category word
			else {
				model.docX[sDoc.index][1]++;
				sDoc.xvec.add(1);
				topic = (int) (Math.random() * model.topicNum);
				sDoc.contents.add(new Pair(word, topic));
				model.categoryTopic[sDoc.prediction][topic]++;
				model.docTopic[sDoc.index][topic]++;
				model.wordTopic[word][topic]++;
				model.numRWords4Cate[sDoc.prediction]++;
				model.numRWords4Topic[topic]++;
			}
		}
	}

	// classification with filtering
	private void fakeSeedWordLoadFilter() {
		LuceneHandler lh = new LuceneHandler(model.luceneIndexPath);
		Document[] lucDocs = lh.getDocs();

		int len = model.docNum;
		if (len == 0) {
			System.out.println("there is no document in the path");
			System.exit(-1);
		}
		SDocument sDoc;
		Document luceneDoc;

		for (int i = 0; i < len; i++) {
			model.documents[i] = new SDocument();
			luceneDoc = lucDocs[i];
			sDoc = model.documents[i];

			sDoc.index = i;
			model.kappa[i] = 0.5;
			sDoc.y = 1;
			sDoc.scores = new double[model.cateNum];
			sDoc.title = luceneDoc.get(LuceneHandler.TITLE);
			sDoc.groundTruth = Integer.parseInt(luceneDoc.get(LuceneHandler.CATE));
			sDoc.check = Integer.parseInt(luceneDoc.get(LuceneHandler.CHECK)) == 1 ? true : false;

			sDoc.xvec = new ArrayList<>();
			if (sDoc.check)
				model.groundTruth[sDoc.groundTruth]++;
			fakeSeedWordInitCont(luceneDoc.get(LuceneHandler.ABSTRACT), sDoc);
		}
	}

	// classification with filtering
	private void fakeSeedWordInitCont(String cont, SDocument sDoc) {

		String[] ves = cont.split(" ");
		double[] re = new double[model.cateNum];

		for (String s : ves) {
			for (int cate = 0; cate < model.iCateNum; cate++)
				if (model.seedSet[cate].contains(s))
					re[cate]++;
			for (int cate = model.iCateNum; cate < model.cateNum; cate++)
				if (model.fakeSeedSet[cate].contains(s))
					re[cate]++;
		}
		initEta(re, sDoc.index);
		sDoc.prediction = (int) Math.floor(Math.random() * model.cateNum);
		if (sDoc.prediction >= model.iCateNum) {
			model.bCateDoc[sDoc.prediction]++;
			model.bDocSum++;
			sDoc.y = 0;
		}
		int word, topic;

		for (String s : ves) {
			word = Dictionary.contains(s);
			if (word == -1)
				continue;
			if (sDoc.prediction < model.iCateNum) {
				// category word
				if (model.seedSet[sDoc.prediction].contains(Dictionary.getWord(word)) && Math.random() < 0.5
						|| Math.random() < (model.rho)) {
					sDoc.xvec.add(0);
					model.docX[sDoc.index][0]++;
					sDoc.contents.add(new Pair(word, -1));
					model.categoryWord[sDoc.prediction][word]++;
					model.numCWords4Cate[sDoc.prediction]++;
				}
				// nor category word
				else {
					model.docX[sDoc.index][1]++;
					sDoc.xvec.add(1);
					topic = (int) (Math.random() * model.topicNum);
					sDoc.contents.add(new Pair(word, topic));
					model.categoryTopic[sDoc.prediction][topic]++;
					model.docTopic[sDoc.index][topic]++;
					model.wordTopic[word][topic]++;
					model.numRWords4Cate[sDoc.prediction]++;
					model.numRWords4Topic[topic]++;
				}

			} else {
				// category word
				if (model.fakeSeedSet[sDoc.prediction].contains(Dictionary.getWord(word)) && Math.random() < 0.5
						|| Math.random() < (model.rho)) {
					sDoc.xvec.add(0);
					model.docX[sDoc.index][0]++;
					sDoc.contents.add(new Pair(word, -1));
					model.categoryWord[sDoc.prediction][word]++;
					model.numCWords4Cate[sDoc.prediction]++;
				}
				// nor category word
				else {
					model.docX[sDoc.index][1]++;
					sDoc.xvec.add(1);
					topic = (int) (Math.random() * model.topicNum);
					sDoc.contents.add(new Pair(word, topic));
					model.categoryTopic[sDoc.prediction][topic]++;
					model.docTopic[sDoc.index][topic]++;
					model.wordTopic[word][topic]++;
					model.numRWords4Cate[sDoc.prediction]++;
					model.numRWords4Topic[topic]++;
				}

			}

		}
	}

	private void initEta(double[] raw, int index) {
		int sum = 0;

		for (int i = 0; i < raw.length; i++)
			raw[i] = Math.log(1 + raw[i]);

		for (double d : raw)
			sum += d;
		if (sum == 0) {
			for (int i = 0; i < raw.length; i++)
				raw[i] = soomth;
		} else {
			for (int i = 0; i < raw.length; i++)
				raw[i] = raw[i] / sum;
		}
		model.eta[index] = raw;
	}

	private void initPhi() {
		for (int c = 0; c < model.cateNum; c++)
			for (int k = 0; k < model.topicNum; k++) {
				model.phi[c][k] = (model.categoryTopic[c][k] + model.alpha0)
						/ (model.numRWords4Cate[c] + model.topicNum * model.alpha0);
				model.phi[c][k] *= model.alpha2;
				model.phiSum[c] += model.phi[c][k];
			}
	}

	public SModel decorateSModel() {
		if (model.method == 0) {
			System.out.println("loading documents...");
			load();
			initPhi();
		} else if (model.method == 1) {
			System.out.println("loading documents...");
			fakeSeedWordLoadFilter();
			initPhi();

		}
		return model;
	}
}
