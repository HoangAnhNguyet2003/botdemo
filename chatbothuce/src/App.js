import React, { useState, useRef, useEffect } from 'react';
import robot_img from "./assets/iconchat.png";
import user_img from "./assets/iconuser.png";
import './App.css';

function App() {
  const messagesEndRef = useRef(null);
  const [query, setQuery] = useState('');
  const [responses, setResponses] = useState([
    {
      type: "start",
      content: "Xin chào! Tôi là HUCE chatbot. Bạn muốn biết thêm thông tin gì hay đặt câu hỏi cho tôi?",
    },
  ]);
  const [isGen, setIsGen] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // Trạng thái mới

  useEffect(() => {
    messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [responses]);

  const handleInputChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    if (event) event.preventDefault();
    if (query.trim() === "") return;

    setIsGen(true);
    setResponses((prev) => [...prev, { type: "end", content: query }]);
    const currentQuery = query;
    setQuery(""); // Reset the input
    setIsLoading(true); // Bắt đầu loading

    try {
      const response = await fetch('http://127.0.0.1:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: currentQuery }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setResponses((prev) => [
        ...prev,
        { type: "start", content: data.response || 'Đã xảy ra lỗi khi lấy câu trả lời.' },
      ]);
    } catch (error) {
      console.error('Error fetching response:', error.message);
      setResponses((prev) => [
        ...prev,
        { type: "start", content: 'Đã xảy ra lỗi khi lấy câu trả lời.' },
      ]);
    } finally {
      setIsLoading(false); // Kết thúc loading
      setIsGen(false);
    }
  };

  const handlePopularQuestionClick = (question) => {
    setQuery(question);
  };

  const popularQuestions = [
    "điều kiện được xét tốt nghiệp",
    "điều kiện đạt tốt nghiệp loại Giỏi",
    "trường xây dựng có thư viện không",
    "thông tin về đội ngũ cán bộ giảng viên của trường",
    "thông tin về ngành Khoa học máy tính",
    "điểm trúng tuyển ngành công nghệ thông tin 2021",
    "thông tin về ký túc xá xây dựng"
  ];

  return (
    <div className="app-container">
      <div className="chatbot-container">
        <div className="chat-window">
          {responses.map((item, index) => (
            <div key={index} className={`chat-message ${item.type === "start" ? "bot" : "user"}`}>
              {item.type === "start" ? (
                <>
                  <img src={robot_img} alt="Bot" className="avatar" />
                  <div className="bot-message">{item.content}</div>
                </>
              ) : (
                <>
                  <div className="user-message">{item.content}</div>
                  <img src={user_img} alt="User" className="avatar" />
                </>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="chat-message bot">
              <img src={robot_img} alt="Bot" className="avatar" />
              <div className="bot-message">Đang nhập dữ liệu...</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="input-area">
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            placeholder="Nhập câu hỏi tại đây..."
            required
            disabled={isGen}
          />
          <button type="submit" disabled={isGen}>Gửi</button>
        </form>
        <p className="note">
          <b>Lưu ý: </b>Mô hình có thể đưa ra câu trả lời không chính xác ở một số trường hợp, hãy kiểm chứng thông tin bạn nhé!
        </p>
      </div>
      <div className="popular-questions">
        <h3>Câu hỏi phổ biến</h3>
        <ul>
          {popularQuestions.map((question, index) => (
            <li key={index} onClick={() => handlePopularQuestionClick(question)}>
              {question}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;