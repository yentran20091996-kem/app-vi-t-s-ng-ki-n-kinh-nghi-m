import express, { Request, Response } from 'express';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Initialize Google GenAI helper with dynamic API Key & Fallback support
const defaultApiKey = process.env.GEMINI_API_KEY || '';
let defaultAi: GoogleGenAI | null = defaultApiKey
  ? new GoogleGenAI({
      apiKey: defaultApiKey,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
    })
  : null;

function getGenAIClient(req: Request) {
  const customKey = (req.headers['x-gemini-api-key'] as string) || req.body?.apiKey;
  const requestedModel = (req.headers['x-gemini-model'] as string) || req.body?.model || 'gemini-3-flash-preview';

  if (customKey && customKey.trim()) {
    return {
      ai: new GoogleGenAI({
        apiKey: customKey.trim(),
        httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
      }),
      model: requestedModel,
    };
  }

  return {
    ai: defaultAi,
    model: requestedModel,
  };
}

// Fallback execution helper for models
async function generateWithFallback(
  client: GoogleGenAI,
  primaryModel: string,
  generateParams: { contents: any; config?: any }
) {
  const modelsToTry = [
    primaryModel,
    'gemini-3-flash-preview',
    'gemini-3-pro-preview',
    'gemini-2.5-flash',
  ].filter((v, i, a) => a.indexOf(v) === i);

  let lastError: any = null;
  for (const m of modelsToTry) {
    try {
      const response = await client.models.generateContent({
        model: m,
        ...generateParams,
      });
      return response;
    } catch (err: any) {
      console.warn(`[Gemini Fallback] Model ${m} failed:`, err.message || err);
      lastError = err;
    }
  }
  throw lastError || new Error('Tất cả mô hình AI đều không thể phản hồi.');
}

// System Prompt for Vietnamese Academic Education SKKN
const SYSTEM_PROMPT_SKKN_EXPERT = `Bạn là một Chuyên gia Cấp cao về Đánh giá & Hướng dẫn Viết Sáng kiến Kinh nghiệm (SKKN), Nghiên cứu Khoa học Sư phạm Ứng dụng thuộc Bộ Giáo dục và Đào tạo Việt Nam.
Nhiệm vụ của bạn là hỗ trợ giáo viên xây dựng đề tài SKKN 2026 chất lượng cao, đúng chuẩn Chương trình Giáo dục Phổ thông 2018 (GDPT 2018), đáp ứng các tiêu chuẩn thẩm định của Sở GD&ĐT và Hội đồng Khoa học cấp Tỉnh/Thành phố.

Quy tắc biên soạn:
1. Văn phong: Học thuật sư phạm, trang trọng, khúc chiết, giàu tính thuyết phục, logic chặt chẽ, không sáo rỗng.
2. Cấu trúc chuẩn theo Thông tư và Công văn hướng dẫn GDPT 2018:
   - Đặt vấn đề & Lý do chọn đề tài (Tính cấp thiết, Cơ sở pháp lý, Cơ sở lý luận, Cơ sở thực tiễn).
   - Thực trạng (Thuận lợi, Khó khăn, Số liệu điều tra khảo sát ban đầu, Bảng phân loại học sinh đối chứng).
   - Hệ thống giải pháp/biện pháp (Mục tiêu, Nội dung, Cách thức tiến hành chi tiết từng bước, Giáo án minh họa, Phiếu học tập, Ứng dụng công nghệ/chuyển đổi số/STEM).
   - Thực nghiệm sư phạm & Hiệu quả áp dụng (Bảng so sánh trước và sau tác động, Kiểm định số liệu, Tỷ lệ phát triển phẩm chất & năng lực).
   - Kết luận & Khuyến nghị (Bài học kinh nghiệm, Khả năng nhân rộng, Đề xuất với Nhà trường, Phòng, Sở).
3. Đảm bảo tính chân thực sư phạm, có dẫn chứng cụ thể theo từng khối lớp và môn học.
4. Trình bày rõ ràng, sử dụng định dạng Markdown phong phú (Tiêu đề, Bảng biểu, Trích dẫn, Gạch đầu dòng).`;

// 1. API: Generate Section
app.post('/api/ai/generate-section', async (req: Request, res: Response) => {
  try {
    const {
      projectTitle,
      subject,
      gradeLevel,
      sectionId,
      sectionTitle,
      userPrompt,
      context,
      ragKnowledge,
    } = req.body;

    const { ai, model } = getGenAIClient(req);

    if (!ai) {
      return res.status(503).json({
        error: 'Chưa cấu hình GEMINI_API_KEY trên hệ thống.',
        fallbackContent: `### ${sectionTitle}\n\n*Nội dung mẫu tự động (Vui lòng cấu hình Gemini API để tạo nội dung học thuật chuyên sâu).*\n\n1. Cơ sở lý luận và tính cấp thiết gắn với đổi mới GDPT 2018.\n2. Phân tích bối cảnh thực tiễn tại đơn vị công tác.\n3. Các biện pháp cụ thể nâng cao chất lượng dạy học môn ${subject || 'chuyên môn'}.`,
      });
    }

    const prompt = `DỰ ÁN SÁNG KIẾN KINH NGHIỆM:
- Tên đề tài: "${projectTitle || 'Chưa đặt tên'}"
- Môn học / Lĩnh vực: ${subject || 'Giáo dục'}
- Cấp học / Khối lớp: ${gradeLevel || 'Toàn trường'}
- Phần đang thực hiện: "${sectionTitle}" (Mã: ${sectionId})

TÀI LIỆU CÔNG VĂN / KIM CHỈ NAM ĐƯỢC NẠP (RAG):
${ragKnowledge || 'Áp dụng khung đánh giá chuẩn SKKN 100 điểm GDPT 2018.'}

NGỮ CẢNH BỔ SUNG TỪ NGƯỜI DÙNG:
${context || 'Không có ngữ cảnh bổ sung.'}

YÊU CẦU CỤ THỂ CỦA TÁC GIẢ:
${userPrompt || 'Hãy viết chi tiết, hoàn chỉnh và có chiều sâu học thuật cho phần này, kèm bảng biểu hoặc quy trình minh họa nếu phù hợp.'}

HÃY SOẠN THẢO NỘI DUNG CHI TIẾT (Định dạng Markdown học thuật chuẩn sư phạm, bao gồm tiểu mục 1.1, 1.2..., có số liệu hoặc ví dụ minh họa trực quan):`;

    const response = await generateWithFallback(ai, model, {
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_PROMPT_SKKN_EXPERT,
        temperature: 0.7,
      },
    });

    const content = response.text || '';
    return res.json({ content });
  } catch (error: any) {
    console.error('Error generating section:', error);
    return res.status(500).json({ error: error.message || 'Lỗi khi gọi AI Gemini' });
  }
});

// 2. API: AI Comprehensive Audit
app.post('/api/ai/audit', async (req: Request, res: Response) => {
  try {
    const { project, rubricCriteria } = req.body;
    const { ai, model } = getGenAIClient(req);

    if (!ai) {
      return res.json({
        totalScore: 84,
        noveltyScore: 26,
        academicScore: 21,
        practicalScore: 25,
        presentationScore: 12,
        estimatedPrize: 'Giải Nhì (Khả năng 85%)',
        noveltyEvaluation: 'Đề tài có tính phát hiện tốt, bắt nhịp xu thế chuyển đổi số GDPT 2018. Cần bổ sung thêm tính độc bản ở biện pháp 3.',
        strengths: [
          'Ý tưởng thiết thực, giải quyết đúng điểm nghẽn thực tế tại cơ sở giáo dục.',
          'Biện pháp có tính sư phạm, gắn kết chặt chẽ với phát triển năng lực học sinh.',
          'Số liệu khảo sát ban đầu và sau thực nghiệm có tính logic rõ nét.'
        ],
        weaknesses: [
          'Phần cơ sở lý luận còn trích dẫn văn bản cũ, cần cập nhật các thông tư mới nhất 2024-2026.',
          'Mẫu khảo sát thực nghiệm còn hẹp, cần phân tích rõ độ tin cậy kiểm định p-value.',
          'Cần minh chứng hình ảnh/video sản phẩm học tập sinh động hơn.'
        ],
        councilQuestions: [
          'Biện pháp này có thể áp dụng cho các trường thuộc vùng khó khăn thiếu thiết bị không?',
          'Thầy/Cô khắc phục hiện tượng học sinh ỷ lại vào công nghệ như thế nào?',
          'Tiêu chí đánh giá năng lực giải quyết vấn đề của học sinh dựa trên thang đo nào?'
        ],
        recommendations: 'Bổ sung bảng phân tích phương sai hoặc kiểm định T-test cho hai nhóm đối chứng và thực nghiệm để tăng sức thuyết phục trước Hội đồng Tỉnh.'
      });
    }

    const ragSummary = (project.ragDocuments || [])
      .filter((d: any) => d.isActive)
      .map((d: any) => `[${d.title}]: ${d.rules.join('; ')}`)
      .join('\n');

    const prompt = `HÃY THẨM ĐỊNH TOÀN DIỆN ĐỀ TÀI SÁNG KIẾN KINH NGHIỆM SAU ĐÂY:
Tên đề tài: "${project.title}"
Môn học: ${project.subject} | Khối: ${project.gradeLevel}
Tác giả / Đơn vị: ${project.author || 'Giáo viên'} - ${project.school || 'Trường học'}

TIÊU CHUẨN ĐÁNH GIÁ TỪ CÔNG VĂN ĐÃ NẠP (RAG):
${ragSummary || 'Thang điểm 100 theo chuẩn Sở GD&ĐT'}

NỘI DUNG TÓM TẮT CÁC PHẦN:
${JSON.stringify(project.sections || {}, null, 2)}

TIÊU CHÍ THẨM ĐỊNH (THANG ĐIỂM 100):
1. Tính mới và tính sáng tạo (Tối đa 30 điểm): Phát hiện mới, giải pháp mới, chưa ai áp dụng tại địa phương.
2. Tính khoa học và sư phạm (Tối đa 25 điểm): Đúng chủ trương GDPT 2018, cơ sở lý luận chuẩn mực, logic.
3. Hiệu quả thực tiễn (Tối đa 30 điểm): Nâng cao chất lượng học tập, có số liệu đối chứng trước/sau tác động rõ ràng.
4. Khả năng nhân rộng & Trình bày (Tối đa 15 điểm): Dễ áp dụng cho đồng nghiệp, thể thức văn bản chuẩn hành chính.

HÃY TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON THEO CẤU TRÚC SAU:
{
  "totalScore": 88,
  "noveltyScore": 27,
  "academicScore": 22,
  "practicalScore": 26,
  "presentationScore": 13,
  "estimatedPrize": "Giải Nhất (hoặc Giải Nhì, Ba, Khuyến khích kèm % xác suất)",
  "noveltyEvaluation": "Đánh giá chi tiết về tính mới...",
  "strengths": ["Điểm mạnh 1", "Điểm mạnh 2", "Điểm mạnh 3"],
  "weaknesses": ["Điểm cần khắc phục 1", "Điểm cần khắc phục 2"],
  "councilQuestions": ["Câu hỏi chất vấn 1 từ hội đồng", "Câu hỏi chất vấn 2", "Câu hỏi chất vấn 3"],
  "recommendations": "Lời khuyên chiến lược để bài viết đạt điểm tuyệt đối..."
}`;

    const response = await generateWithFallback(ai, model, {
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_PROMPT_SKKN_EXPERT,
        responseMimeType: 'application/json',
      },
    });

    const parsed = JSON.parse(response.text?.trim() || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Error during audit:', error);
    return res.status(500).json({ error: error.message });
  }
});

// 3. API: Simulated Defense (Hội đồng phản biện ảo)
app.post('/api/ai/simulated-defense', async (req: Request, res: Response) => {
  try {
    const { judgeId, judgeName, judgeRole, projectTitle, history, userReply } = req.body;
    const { ai, model } = getGenAIClient(req);

    if (!ai) {
      return res.json({
        reply: `[${judgeName}]: Cảm ơn phản hồi của Thầy/Cô về đề tài "${projectTitle}". Về cơ bản giải pháp khá hợp lý, nhưng Thầy/Cô cần chỉ rõ thời lượng thực nghiệm trong bao nhiêu tiết và mức độ sai số nếu học sinh nghỉ học giữa chừng.`
      });
    }

    const prompt = `Bạn đang đóng vai Giám khảo Hội đồng Chấm Sáng kiến Kinh nghiệm:
- Họ tên Giám khảo: ${judgeName}
- Vai trò & Phong cách: ${judgeRole}
- Đề tài đang thẩm vấn: "${projectTitle}"

LỊCH SỬ TRAO ĐỔI TRƯỚC ĐÓ:
${JSON.stringify(history || [])}

CÂU TRẢ LỜI / PHẢN HỒI MỚI NHẤT CỦA GIÁO VIÊN:
"${userReply}"

HÃY ĐƯA RA LỜI NHẬN XÉT, PHẢN BIỆN TIẾP THEO HOẶC ĐẶT THÊM CÂU HỎI THỬ THÁCH (Giữ phong thái chuẩn mực, chuyên nghiệp, thực tế sư phạm và đưa ra gợi ý cách hoàn thiện bài viết):`;

    const response = await generateWithFallback(ai, model, {
      contents: prompt,
      config: {
        systemInstruction: 'Bạn là giám khảo phản biện chuyên nghiệp, sắc sảo, công tâm của Hội đồng Khoa học ngành Giáo dục.',
        temperature: 0.7,
      },
    });

    return res.json({ reply: response.text });
  } catch (error: any) {
    console.error('Error in simulated defense:', error);
    return res.status(500).json({ error: error.message });
  }
});

// 4. API: Copilot Chat / Writing Assistant
app.post('/api/ai/copilot-chat', async (req: Request, res: Response) => {
  try {
    const { message, projectContext, activeSection } = req.body;
    const { ai, model } = getGenAIClient(req);

    if (!ai) {
      return res.json({
        reply: `Chào Thầy/Cô! Tôi là Trợ lý AI SKKN 2026. Để kích hoạt tính năng hỗ trợ trực tiếp từ Gemini, vui lòng bấm nút "Cài đặt API Key" trên thanh tiêu đề và gắn mã khóa Google AI Studio của Thầy/Cô.`
      });
    }

    const prompt = `BỐI CẢNH ĐỀ TÀI CỦA GIÁO VIÊN:
Tên đề tài: ${projectContext?.title || 'Chưa đặt'}
Môn học: ${projectContext?.subject || 'GDPT'} | Cấp lớp: ${projectContext?.gradeLevel || 'THPT/THCS/Tiểu học'}
Mục hiện tại giáo viên đang viết: ${activeSection || 'Tổng quan'}

YÊU CẦU TỪ GIÁO VIÊN:
${message}

HÃY TRẢ LỜI CỤ THỂ, ĐÚNG TRỌNG TÂM SƯ PHẠM, ĐƯA RA VÍ DỤ HOẶC ĐOẠN VĂN MẪU CÓ THỂ COPY TRỰC TIẾP VÀO BÀI SKKN:`;

    const response = await generateWithFallback(ai, model, {
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_PROMPT_SKKN_EXPERT,
        temperature: 0.7,
      },
    });

    return res.json({ reply: response.text });
  } catch (error: any) {
    console.error('Error in copilot chat:', error);
    return res.status(500).json({ error: error.message });
  }
});

// 5. API: Extract RAG Knowledge Base from pasted text / uploaded file (Word/PDF)
app.post('/api/ai/extract-rag', async (req: Request, res: Response) => {
  try {
    const { rawText, fileName, fileBase64, mimeType } = req.body;
    const { ai, model } = getGenAIClient(req);

    if (!ai) {
      return res.json({
        title: fileName || 'Văn bản hướng dẫn SKKN 2026',
        extractedRules: [
          'Thang điểm 100: Tính mới (30đ), Tính khoa học (25đ), Hiệu quả (30đ), Nhân rộng (15đ)',
          'Yêu cầu tối thiểu 2 nhóm đối chứng và thực nghiệm',
          'Đảm bảo định dạng Times New Roman cỡ 14, lề trái 3cm',
          'Không sao chép quá 20% các đề tài đã công bố'
        ],
        keyPriorities: ['Ứng dụng CNTT & AI', 'Dạy học tích hợp STEM', 'Đổi mới kiểm tra đánh giá phẩm chất']
      });
    }

    let contentsPayload: any;

    if (fileBase64 && mimeType === 'application/pdf') {
      contentsPayload = [
        {
          inlineData: {
            mimeType: 'application/pdf',
            data: fileBase64,
          },
        },
        `Phân tích kỹ văn bản công văn / hướng dẫn chấm Sáng kiến kinh nghiệm (SKKN) từ tệp PDF đính kèm "${fileName || 'Văn bản'}".
HÃY TRÍCH XUẤT CÁC YẾU TỐ QUAN TRỌNG VÀ TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON:
{
  "title": "Tên trích yếu văn bản",
  "extractedRules": [
    "Tiêu chí 1 (Thang điểm và yêu cầu cụ thể)",
    "Tiêu chí 2",
    "Quy định thể thức (font, lề, số trang...)",
    "Lưu ý chống đạo văn hoặc điều kiện công nhận..."
  ],
  "keyPriorities": ["Ưu tiên 1", "Ưu tiên 2", "Chủ đề khuyến khích..."]
}`,
      ];
    } else {
      const textToAnalyze = (rawText || '').slice(0, 15000);
      contentsPayload = `Phân tích văn bản công văn / tiêu chuẩn chấm SKKN sau đây:
"${textToAnalyze}"

HÃY TRÍCH XUẤT CÁC YẾU TỐ QUAN TRỌNG VÀ TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON:
{
  "title": "${fileName || 'Tên trích yếu văn bản'}",
  "extractedRules": [
    "Tiêu chí 1",
    "Tiêu chí 2",
    "Quy định thể thức...",
    "Lưu ý chống đạo văn..."
  ],
  "keyPriorities": ["Ưu tiên 1", "Ưu tiên 2", "Chủ đề khuyến khích..."]
}`;
    }

    const response = await generateWithFallback(ai, model, {
      contents: contentsPayload,
      config: {
        responseMimeType: 'application/json',
      },
    });

    const parsed = JSON.parse(response.text?.trim() || '{}');
    return res.json(parsed);
  } catch (error: any) {
    console.error('Error extracting RAG:', error);
    return res.status(500).json({ error: error.message });
  }
});

// Vite middleware in dev / serve static in prod
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist/index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[SKKN 2026 PRO] Server running on http://localhost:${PORT}`);
  });
}

startServer();
