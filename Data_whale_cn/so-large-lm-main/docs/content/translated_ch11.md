# Chapter 11 Big Model Law

## 11.1 Introduction

In this tutorial, we will explore what the law says about the development and deployment of large language models. We will proceed as follows:

1. **How ​​the New Technology Relates to Existing Law**

As with our previous lectures, such as the lecture on social bias, much of what we will discuss is not necessarily specific to large language models (there is no specific large language model law). However, whenever a new and powerful technology emerges, it raises many questions about whether existing law is still applicable or meaningful. For example, as the importance of the Internet has increased, [Internet law](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3191751) (or cyber law) has emerged. It draws on knowledge from existing fields such as intellectual property law, privacy law, and contract law.

2. **Unique Challenges of the Internet**

Laws are usually clearly defined by jurisdiction (e.g., state, federal), but the Internet is not limited by geography. On the Internet, you can remain anonymous, and anyone can post a piece of content that can theoretically be viewed by anyone.

3. **The difference between law and morality**

Law can be enforced by the government, while morality cannot be enforced and can be created by any organization. For example, [the Hippocratic Oath of a Physician](https://en.wikipedia.org/wiki/Hippocratic_Oath), [ACM's Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics), [NeurIPS's Code of Conduct](https://neurips.cc/public/CodeOfConduct), and [Stanford's Honor Code, etc.](https://communitystandards.stanford.edu/policies-and-guidance/honor-code).

4. **Jurisdictional issues of law**

Depending on where you are (which country, which state, etc.), the applicable law will be different. For example, the EU's data privacy law [GDPR](https://gdpr-info.eu/) is more comprehensive than the US law. Laws can exist at the federal, state, or local level.

5. **Types of law**

Common types of law include common law (judicial), statutory law (legislative), and regulatory law (administrative).

6. **Large language models**

We will turn our attention to large language models. Recall the life cycle of a large language model: collect training data, train a large language model, adapt it to downstream tasks, deploy it to users, andDeployment of language models.

There are two main areas where the law intersects in the lifecycle of large language models: data and applications.

7. **Data**

All machine learning relies on data. Language models rely on large amounts of data, especially data of others produced for other purposes, which is often scraped without permission. Intellectual property law protects the creators of data, so does training a language model on that data constitute copyright infringement? Privacy law protects the privacy rights of individuals, so is training a language model on public or private data a potential privacy violation? For private data, when can it be collected and aggregated?

8. **Application**

Language models can be used for a wide range of downstream tasks (e.g., question answering, chatbots). The technology may be used intentionally for harm (e.g., spam, phishing attacks, harassment, fake news). Existing internet fraud and abuse laws may cover some of these. They can be deployed in a variety of high-risk settings (e.g., healthcare, lending, education). Existing regulations in related fields (e.g., healthcare) may cover some of these.

The ability of large language models to scale (e.g., realistic text generation, chatbots) will bring new challenges.

## 11.2 Copyright Law

Large language models, or any machine learning models, are trained on data that is the result of human labor (e.g., authors, programmers, photographers, etc.).). What use others can make of these creations (e.g., books, code, photos, etc.) besides the creator falls under the purview of intellectual property law.

### 11.2.1 Intellectual Property Law

The motivation is to encourage the creation of all types of intellectual products. If anyone can take advantage of your hard work and profit from it, people will lose the motivation to create or share. Intellectual property includes: copyright, patents, trademarks, trade secrets.

In the United States, the key statute that determines copyright is the Copyright Act of 1976. Copyright protection applies to "original works of authorship that have been fixed in some tangible medium that can be perceived, reproduced, or otherwise communicated directly or by machine or device." The Copyright Act of 1976 expanded the scope of copyright protection from "published" (1909) to "fixed." Although registration is not required to obtain copyright protection, creators must register their copyrights before suing others for copyright infringement. Copyright protection lasts for 75 years, after which the copyright expires and the work becomes part of the public domain (such as the works of Shakespeare, Beethoven, etc.).

There are two ways to use a copyrighted work: get a license or rely on fair use terms.

### 11.2.2 Licenses

A license (from contract law) is granted by the licensor to the licensee. In effect, "a license is a promise not to sue". Creative Commons licenses allow free distribution of copyrighted works. [For example](https://en.wikipedia.org/wiki/List_of_major_Creative_Commons_licensed_works), Wikipedia, OpenCourseWare, Khan Academy, Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.

### 11.2.3 Fair Use (Section 107)

Fair use has been common law since the 1840s. Four factors determine whether fair use applies:

1. The purpose and nature of the use (educational use over commercial use, transformative use over reproduction);

2. The nature of the copyrighted work (fiction over factual work, degree of novelty);

3. The amount and substantiality of the original work used; and

4. The effect of the use on the market (or potential market) for the original work.

Terms of Service may

add additional restrictions. For example, YouTube's Terms of Service prohibit downloading videos, even if they are licensed under Creative Commons.

Note: Facts and ideas are not copyrightable. A database of facts can be copyrightable if the curation/arrangement is considered expression. Copying data (the first step in training) is already an infringement, even if you don't do anything else. Statutory damages can be as high as $150,000 per work (Section 504 of the Copyright Act).

## 11.3 Case Studies

Next, we'll review some cases where fair use has been ruled for or against fair use.### 11.3.1 [Authors Guild v. Google](https://en.wikipedia.org/wiki/Authors_Guild,_Inc._v._Google,_Inc.)

Google Book Search, which scanned printed books and made them searchable online (showing snippets), began in 2002. The Authors Guild complained that Google did not seek their permission for books that were still under copyright. In 2013, a district court ruled that Google was fair use.

### 11.3.2 [Google v. Oracle](https://en.wikipedia.org/wiki/Google_LLC_v._Oracle_America,_Inc.)

Google copied 37 Java APIs owned by Oracle (formerly Sun Microsystems) in its Android operating system. Oracle sued Google for copyright infringement. In April 2021, the Supreme Court ruled that Google's use of the Java APIs was fair use.

### 11.3.3 [Fox News v. TVEyes](https://www.lexisnexis.com/community/casebrief/p/casebrief-fox-news-network-llc-v-tveyes-inc)

TVEyes records TV shows and creates a service that lets people search (via text) and watch 10-second clips. Fox News sues TVEyes. In 2018, the Second District rules in favor of Fox News, finding it not fair use.

### 11.3.4 [Kelly v. Arriba](https://en.wikipedia.org/wiki/Kelly_v._Arriba_Soft_Corp.)

Arriba creates a search engine that displays thumbnails. Kelly (individual) sues Arriba. In 2003, the Ninth Circuit rules in favor of Arriba, finding it fair use.

### 11.3.5 [Sega v. Accolade](https://en.wikipedia.org/wiki/Sega_v._Accolade)

In 1989, the Sega Genesis video game console is released. Accolade wanted to publish the game on the Genesis, but Sega charged an extra fee in exchange for exclusive distribution. Accolade reverse-engineered Sega's code to create a new version that bypassed the security lock. Sega sued Accolade in 1991. In 1992, the Ninth CircuitThe court ruled in favor of Accolade, finding it fair use.

## 11.4 Fair Learning vs. Machine Learning

Fair Learning argues that machine learning is fair use. The use of data by machine learning systems is transformative, it does not change the work, but it changes the purpose. Machine learning systems are interested in ideas, not specific expressions.

Arguments for machine learning as fair use: Wide access to training data creates better systems for society. If use is not allowed, then most works cannot be used to generate new value. Using copyrighted data may be fairer.

Arguments against machine learning as fair use: Arguments that machine learning systems do not produce creative "end products" but simply make money. Generative models (e.g., language models) can compete with creative professionals. Problems with machine learning systems (spreading false information, enabling surveillance, etc.), so machine learning systems' benefits should not be given the benefit of the doubt.

Under copyright law, it is difficult to separate what is protectable (e.g., expression) from what is not protectable (e.g., ideas). While there are many reasons why building machine learning systems may be wrong, is copyright the right tool to stop it? The question of whether training large language models is fair use is developing rapidly.

## 11.5 Conclusion of the stages

Looking at the history of information technology, we can see three stages:

1. The first stage: text data mining (search engines), based on simple pattern matching.

2. The second stage: classification (e.g., classification stop signs3. Phase 3: Generative Models that Learn to Imitate Expressions.

Last time, we saw that extracting training data from GPT-2 can present privacy issues. If a language model directly copies Harry Potter, then this is problematic for fair use. However, even if a language model does not directly generate previous works, copyright is still relevant because previous copyrighted works were used to train the language model.

In fact, a language model can compete with an author. For example, a writer writes 3 books, and a language model is trained on these 3 books and automatically generates a 4th.

Therefore, the future of copyright and machine learning in the face of large language models is unknown.

## 11.6 Privacy Law Tutorial

In this tutorial, we will briefly discuss some examples of privacy laws, including Clearview AI, the California Consumer Privacy Act (2018), the California Privacy Rights Act (2020), and the European Union's General Data Protection Regulation (GDPR).

### 11.6.1 [Clearview AI](https://en.wikipedia.org/wiki/Clearview_AI)

Clearview AI is a company founded in 2017. In 2019, the New York Times exposed it. By October 2021, the company had acquiredbe, Venmo, and other websites scraped 10 billion face images. The company sold the data to law enforcement agencies (e.g., the FBI) ​​and commercial organizations. The company argued that it had the right to use the publicly available information. The company has been sued for privacy violations.

### 11.6.2 [Illinois Biometric Information Privacy Act (2008)](https://en.wikipedia.org/wiki/California_Privacy_Rights_Act)

This law regulates biometric identifiers by private entities (not including government entities). Clearview deleted the Illinois data. The EU Hamburg Data Protection Authority (DPA) found the action illegal.

### 11.6.3 [California Consumer Privacy Act (2018)](https://en.wikipedia.org/wiki/California_Consumer_Privacy_Act)

This law gives California residents the following rights:
- Know what personal data is collected about them.
- Know if their personal data has been sold or disclosed, and to whom.
- Opt out of the sale of their personal data.
- Access their personal data.
- Request that a business delete any personal information collected from consumers.
- Not be discriminated against for exercising their privacy rights.

Personal data includes: real name, alias, mailing address, unique personal identifier, online identifier, IP address, email address, account name, social security number, driver's license number, license plate number, passport number, etc.

The law applies to businesses operating in California with annual revenue of at least $25 million. There is no federal equivalent in the United States. Unlike GDPR, this law does not allow users to correct data.

### 11.6.4 [California Privacy Rights Act (2020)](https://en.wikipedia.org/wiki/California_Privacy_Rights_Act)

This bill creates the California Privacy Shield, which will take effect on January 1, 2023 and apply to data collected after January 1, 2022.

#### 11.6.4.1 Intent

- Understand who is collecting their and their children's personal information, how it is used, and to whom it is disclosed.

- Control the use of their personal information, including
- Limit the use of their sensitive personal information.
- Access their personal information and have the ability to correct, delete, and transfer their personal information.
- Exercise their privacy rights through easily accessible self-service tools.
- Exercise their privacy rights without penalty.
- Hold businesses accountable for not taking reasonable information security precautions.
- Benefit from businesses using their personal information.
- As an employeeand independent contractors can also protect their privacy interests.

## 11.7 [GDPR (General Data Protection Regulation)](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation)

This regulation is part of EU law on data privacy, passed in 2016 and enforceable in 2018. It has a wider scope than CCPA. It does not apply to national security activities or law enforcement actions involving the processing of personal data. Data subjects can consent to the processing of personal data and can withdraw it at any time. People should have the right to access their personal data. Google was fined $57 million for not obtaining consent for ad personalization during the setup process of Android phones.

## 11.8 [Other Laws](https://www.natlawreview.com/article/california-s-bot-disclosure-law-sb-1001-now-effect)

### 11.8.1 California's Bot Disclosure Act:

It is illegal to use a bot to communicate with a person without disclosing that it is a bot. Limitations: Only for incentivizing sales or influencing election votes. Limitations: Only for public websites with 10 million visitors per month in the US.

## 11.9 Summary

InWhen we train large language models, we have to face the question of copyright and fair use. Due to the unfiltered nature of web crawling, you have to appeal to fair use (getting licenses from everyone will be very difficult). The generative nature of the model may make it challenging to argue fair use (can compete with humans). At what level does it make sense to regulate (language model or downstream applications)? This field is rapidly evolving and requires deep legal and AI expertise to make an informed decision!

## Further reading

- [Foundation models report (legality section)](https://crfm.stanford.edu/assets/report.pdf#legality)
- [AI Regulation is coming](https://hbr.org/2021/09/ai-regulation-is-coming)
- [Fair Learning](https://texaslawreview.org/fair-learning/). _Mark Lemley, Bryan Casey_. Texas Law Review, 2021.
- [You might be a robot](https://www.cornelllawreview.org/wp-content/uploads/2020/07/Casey-Lemley-final-2.pdf)